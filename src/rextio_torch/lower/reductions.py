"""Lower ``.mean`` / ``.sum`` claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.reductions import MEAN_RULE, REDUCTION_RULES, SUM_RULE
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.rust_snippets import (
    MEAN_DIM1_KEEPFALSE,
    SUM_DIM1_KEEPFALSE,
    boundary_helpers,
    mean_dim1_keepfalse_helper,
    sum_dim1_keepfalse_helper,
)

_HELPERS: dict[str, tuple[str, str]] = {
    MEAN_RULE: (MEAN_DIM1_KEEPFALSE, mean_dim1_keepfalse_helper()),
    SUM_RULE: (SUM_DIM1_KEEPFALSE, sum_dim1_keepfalse_helper()),
}

_METHOD_BY_RULE: dict[str, str] = {
    MEAN_RULE: "mean",
    SUM_RULE: "sum",
}


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed mean/sum method site, or return None."""
    method = _method_name(claimed.target)
    if claimed.kind != "call" or method not in {"mean", "sum"}:
        return None
    if claimed.rule_id not in REDUCTION_RULES:
        raise ValueError(
            "rextio-torch reduction lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    if _METHOD_BY_RULE.get(claimed.rule_id or "") != method:
        raise ValueError(
            "rextio-torch reduction lower rule/method mismatch: "
            f"rule_id={claimed.rule_id!r} method={method!r}"
        )
    receiver = claimed.receiver
    if (
        receiver is None
        or receiver.arg_type != TENSOR_F32_CPU_2D
        or claimed.operand_types
        or claimed.result_type != TENSOR_F32_CPU_1D
    ):
        raise ValueError(f"rextio-torch received malformed {method} lower metadata")
    if len(claimed.keywords) != 2:
        raise ValueError(
            f"rextio-torch {method} lower requires exactly two keywords (dim, keepdim)"
        )
    values = {kw.name: kw.literal for kw in claimed.keywords}
    if set(values) != {"dim", "keepdim"}:
        raise ValueError(
            f"rextio-torch {method} lower keyword names changed: {sorted(values)!r}"
        )
    dim_lit = values["dim"]
    keepdim_lit = values["keepdim"]
    if (
        not dim_lit.is_literal
        or not isinstance(dim_lit.value, int)
        or isinstance(dim_lit.value, bool)
        or dim_lit.value != 1
    ):
        raise ValueError(
            f"rextio-torch {method} lower requires dim=1 literal; got {dim_lit.value!r}"
        )
    if not keepdim_lit.is_literal or keepdim_lit.value is not False:
        raise ValueError(
            f"rextio-torch {method} lower requires keepdim=False literal; "
            f"got {keepdim_lit.value!r}"
        )
    if ctx.receiver is None:
        raise ValueError(f"rextio-torch {method} lower requires ctx.receiver")
    rule_id = claimed.rule_id
    if rule_id is None or rule_id not in _HELPERS:
        raise ValueError(
            f"rextio-torch reduction lower missing helper for rule_id: {rule_id!r}"
        )
    call_name, helper = _HELPERS[rule_id]
    return LoweredExpr(
        rust=f"{call_name}(&{ctx.receiver})?",
        helpers=(boundary_helpers(), helper),
    )


__all__ = ["try_lower"]
