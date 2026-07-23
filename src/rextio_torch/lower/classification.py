"""Lower the fail-closed Torch classification-head method claims."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.classification import ARGMAX_RULE, CLASSIFICATION_RULES, SOFTMAX_RULE
from rextio_torch.diagnostics import TENSOR_F32_CPU_2D, TENSOR_I64_CPU_1D
from rextio_torch.rust_snippets import (
    ARGMAX_DIM1_KEEPFALSE,
    SOFTMAX_DIM1,
    argmax_dim1_keepfalse_helper,
    boundary_helpers,
    softmax_dim1_helper,
)

_EXPECTED: dict[str, tuple[str, frozenset[str], str, str, str]] = {
    SOFTMAX_RULE: (
        "softmax",
        frozenset({"dim"}),
        TENSOR_F32_CPU_2D,
        SOFTMAX_DIM1,
        softmax_dim1_helper(),
    ),
    ARGMAX_RULE: (
        "argmax",
        frozenset({"dim", "keepdim"}),
        TENSOR_I64_CPU_1D,
        ARGMAX_DIM1_KEEPFALSE,
        argmax_dim1_keepfalse_helper(),
    ),
}


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a classification method only after fully revalidating claim metadata."""
    method = _method_name(claimed.target)
    if claimed.kind != "call" or method not in {"softmax", "argmax"}:
        return None
    if claimed.rule_id not in CLASSIFICATION_RULES:
        raise ValueError(
            "rextio-torch classification lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    rule_id = claimed.rule_id
    if rule_id is None:
        raise ValueError("rextio-torch classification lower missing rule_id")
    expected_method, keyword_names, result_type, call_name, helper = _EXPECTED[rule_id]
    if method != expected_method:
        raise ValueError(
            "rextio-torch classification lower rule/method mismatch: "
            f"rule_id={rule_id!r} method={method!r}"
        )
    receiver = claimed.receiver
    if (
        receiver is None
        or receiver.arg_type != TENSOR_F32_CPU_2D
        or claimed.operand_types
        or claimed.result_type != result_type
    ):
        raise ValueError(f"rextio-torch received malformed {method} lower metadata")
    values = {kw.name: kw.literal for kw in claimed.keywords}
    if len(values) != len(claimed.keywords) or set(values) != keyword_names:
        raise ValueError(
            f"rextio-torch {method} lower keyword names changed: {sorted(values)!r}"
        )
    dim = values["dim"]
    if (
        not dim.is_literal
        or not isinstance(dim.value, int)
        or isinstance(dim.value, bool)
        or dim.value != 1
    ):
        raise ValueError(
            f"rextio-torch {method} lower requires dim=1 literal; got {dim.value!r}"
        )
    if method == "argmax":
        keepdim = values["keepdim"]
        if not keepdim.is_literal or keepdim.value is not False:
            raise ValueError(
                "rextio-torch argmax lower requires keepdim=False literal; "
                f"got {keepdim.value!r}"
            )
    if ctx.receiver is None:
        raise ValueError(f"rextio-torch {method} lower requires ctx.receiver")
    return LoweredExpr(
        rust=f"{call_name}(&{ctx.receiver})?",
        helpers=(boundary_helpers(), helper),
    )


__all__ = ["try_lower"]
