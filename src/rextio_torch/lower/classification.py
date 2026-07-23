"""Lower representable static softmax/argmax claims."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.classification import (
    ARGMAX_RULE,
    ARGMAX_STATIC_RULE,
    CLASSIFICATION_RULES,
    SOFTMAX_RULE,
    SOFTMAX_STATIC_RULE,
)
from rextio_torch.diagnostics import (
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_torch.rust_snippets import (
    argmax_call_name,
    argmax_helper,
    boundary_helpers,
    softmax_call_name,
    softmax_helper,
)

_RULE_METHODS: dict[str, str] = {
    SOFTMAX_RULE: "softmax",
    SOFTMAX_STATIC_RULE: "softmax",
    ARGMAX_RULE: "argmax",
    ARGMAX_STATIC_RULE: "argmax",
}
_LEGACY_RULES = frozenset({SOFTMAX_RULE, ARGMAX_RULE})
_FUNCTION_TARGETS: dict[str, str] = {
    "torch.softmax": "softmax",
    "torch.argmax": "argmax",
}
_F32_TYPES = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def _keyword_literals(claimed: ClaimSite) -> dict[str, object]:
    values: dict[str, object] = {}
    for keyword in claimed.keywords:
        if keyword.name in values or not keyword.literal.is_literal:
            raise ValueError(
                "rextio-torch classification lower requires unique literal keywords"
            )
        values[keyword.name] = keyword.literal.value
    return values


def _form_metadata(
    claimed: ClaimSite,
    ctx: LoweringContext,
    method: str,
) -> tuple[str, int | None, bool, str]:
    if claimed.receiver is None:
        if _FUNCTION_TARGETS.get(claimed.target) != method or ctx.receiver is not None:
            raise ValueError(
                f"rextio-torch functional {method} lower received non-canonical metadata"
            )
        base_arity = 1
        if len(claimed.operand_types) not in {1, 2} or len(ctx.operands) != len(
            claimed.operand_types
        ):
            raise ValueError(
                f"rextio-torch functional {method} lower requires tensor plus optional dim"
            )
        input_type = claimed.operand_types[0]
        input_name = ctx.operands[0]
    else:
        if ctx.receiver is None:
            raise ValueError(f"rextio-torch method {method} lower requires ctx.receiver")
        base_arity = 0
        if len(claimed.operand_types) not in {0, 1} or len(ctx.operands) != len(
            claimed.operand_types
        ):
            raise ValueError(
                f"rextio-torch method {method} lower accepts at most one positional dim"
            )
        input_type = claimed.receiver.arg_type
        input_name = ctx.receiver
    if input_type not in _F32_TYPES:
        raise ValueError(
            f"rextio-torch {method} lower requires float32 CPU rank-1/2 input"
        )

    has_positional_dim = len(claimed.operand_types) == base_arity + 1
    positional_dim: int | None = None
    if has_positional_dim:
        if len(claimed.operand_literals) != len(claimed.operand_types):
            raise ValueError(
                f"rextio-torch {method} lower positional literal metadata is not aligned"
            )
        dim_index = base_arity
        literal = claimed.operand_literals[dim_index]
        if (
            claimed.operand_types[dim_index] != "int"
            or not literal.is_literal
            or not isinstance(literal.value, int)
            or isinstance(literal.value, bool)
        ):
            raise ValueError(
                f"rextio-torch {method} lower positional dim is not a proved int literal"
            )
        positional_dim = literal.value
    return input_type, positional_dim, has_positional_dim, input_name


def _expected_result(
    method: str,
    input_type: str,
    dim: int,
    keepdim: bool,
) -> str | None:
    if method == "softmax":
        if input_type == TENSOR_F32_CPU_1D and dim == 0:
            return TENSOR_F32_CPU_1D
        if input_type == TENSOR_F32_CPU_2D and dim in {0, 1}:
            return TENSOR_F32_CPU_2D
        return None
    if input_type == TENSOR_F32_CPU_2D and dim in {0, 1} and not keepdim:
        return TENSOR_I64_CPU_1D
    if input_type == TENSOR_F32_CPU_1D and dim == 0 and keepdim:
        return TENSOR_I64_CPU_1D
    return None


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower classification calls after independent form/literal/type checks."""
    method = _method_name(claimed.target)
    if claimed.kind != "call" or method not in {"softmax", "argmax"}:
        return None
    if claimed.rule_id not in CLASSIFICATION_RULES:
        raise ValueError(
            "rextio-torch classification lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    if _RULE_METHODS.get(claimed.rule_id or "") != method:
        raise ValueError(
            "rextio-torch classification lower rule/method mismatch: "
            f"rule_id={claimed.rule_id!r} method={method!r}"
        )

    input_type, positional_dim, has_positional_dim, input_name = _form_metadata(
        claimed, ctx, method
    )
    values = _keyword_literals(claimed)
    allowed = set() if has_positional_dim else {"dim"}
    if method == "argmax":
        allowed.add("keepdim")
    if not set(values) <= allowed:
        raise ValueError(
            f"rextio-torch {method} lower received duplicate/unsupported options"
        )
    raw_dim: object
    if has_positional_dim:
        raw_dim = positional_dim
    else:
        if "dim" not in values:
            raise ValueError(f"rextio-torch {method} lower requires a static dim")
        raw_dim = values["dim"]
    keepdim = values.get("keepdim", False)
    if (
        not isinstance(raw_dim, int)
        or isinstance(raw_dim, bool)
        or raw_dim not in {0, 1}
        or not isinstance(keepdim, bool)
    ):
        raise ValueError(
            f"rextio-torch {method} lower requires dim 0/1 and bool keepdim"
        )
    dim = raw_dim
    if method == "softmax" and "keepdim" in values:
        raise ValueError("rextio-torch softmax lower does not accept keepdim")

    expected_result = _expected_result(method, input_type, dim, keepdim)
    if expected_result is None or claimed.result_type != expected_result:
        raise ValueError(
            f"rextio-torch {method} lower result metadata changed between claim and lower"
        )

    if claimed.rule_id in _LEGACY_RULES:
        if not (
            claimed.receiver is not None
            and not has_positional_dim
            and input_type == TENSOR_F32_CPU_2D
            and dim == 1
            and (
                (method == "softmax" and set(values) == {"dim"})
                or (
                    method == "argmax"
                    and keepdim is False
                    and set(values) == {"dim", "keepdim"}
                )
            )
        ):
            raise ValueError(
                f"rextio-torch legacy {method} lower metadata changed between claim and lower"
            )
    elif claimed.rule_id not in {SOFTMAX_STATIC_RULE, ARGMAX_STATIC_RULE}:
        raise ValueError(
            f"rextio-torch classification lower missing static rule: {claimed.rule_id!r}"
        )

    if method == "softmax":
        call_name = softmax_call_name(dim)
        helper = softmax_helper(dim)
    else:
        call_name = argmax_call_name(dim, keepdim)
        helper = argmax_helper(dim, keepdim, expected_rank=1)
    return LoweredExpr(
        # Positional dim remains compile-time metadata and is never forwarded
        # to the runtime tensor helper.
        rust=f"{call_name}(&{input_name})?",
        helpers=(boundary_helpers(), helper),
    )


__all__ = ["try_lower"]
