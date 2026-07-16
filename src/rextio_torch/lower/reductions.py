"""Lower ``.mean(dim=1, keepdim=False)`` claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.reductions import MEAN_RULE
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.rust_snippets import (
    MEAN_DIM1_KEEPFALSE,
    boundary_helpers,
    mean_dim1_keepfalse_helper,
)


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed mean method site, or return None."""
    if claimed.kind != "call" or _method_name(claimed.target) != "mean":
        return None
    if claimed.rule_id != MEAN_RULE:
        raise ValueError(
            "rextio-torch mean lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {MEAN_RULE!r}"
        )
    receiver = claimed.receiver
    if (
        receiver is None
        or receiver.arg_type != TENSOR_F32_CPU_2D
        or claimed.operand_types
        or claimed.result_type != TENSOR_F32_CPU_1D
    ):
        raise ValueError("rextio-torch received malformed mean lower metadata")
    if len(claimed.keywords) != 2:
        raise ValueError(
            "rextio-torch mean lower requires exactly two keywords (dim, keepdim)"
        )
    values = {kw.name: kw.literal for kw in claimed.keywords}
    if set(values) != {"dim", "keepdim"}:
        raise ValueError(
            f"rextio-torch mean lower keyword names changed: {sorted(values)!r}"
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
            f"rextio-torch mean lower requires dim=1 literal; got {dim_lit.value!r}"
        )
    if not keepdim_lit.is_literal or keepdim_lit.value is not False:
        raise ValueError(
            "rextio-torch mean lower requires keepdim=False literal; "
            f"got {keepdim_lit.value!r}"
        )
    if ctx.receiver is None:
        raise ValueError("rextio-torch mean lower requires ctx.receiver")
    return LoweredExpr(
        rust=f"{MEAN_DIM1_KEEPFALSE}(&{ctx.receiver})?",
        helpers=(boundary_helpers(), mean_dim1_keepfalse_helper()),
    )


__all__ = ["try_lower"]
