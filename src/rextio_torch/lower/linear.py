"""Lower functional linear claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.linear import LINEAR_RULE, LINEAR_TARGET
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.rust_snippets import LINEAR, boundary_helpers, linear_helper

_EXPECTED = (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D)


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed functional linear site, or return None."""
    if claimed.kind != "call" or claimed.target != LINEAR_TARGET:
        return None
    if claimed.rule_id != LINEAR_RULE:
        raise ValueError(
            "rextio-torch linear lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {LINEAR_RULE!r}"
        )
    if claimed.keywords or len(claimed.operand_types) != 3:
        raise ValueError(
            "rextio-torch linear lower requires three positional operands and no keywords"
        )
    if tuple(claimed.operand_types) != _EXPECTED:
        raise ValueError(
            "rextio-torch linear lower operand types changed between claim and lower: "
            f"{claimed.operand_types!r}"
        )
    if claimed.result_type != TENSOR_F32_CPU_2D:
        raise ValueError(
            "rextio-torch linear lower result type changed between claim and lower: "
            f"{claimed.result_type!r}"
        )
    if len(ctx.operands) != 3:
        raise ValueError(
            "rextio-torch linear lower requires three ctx.operands entries; "
            f"got {len(ctx.operands)}"
        )
    a, b, c = ctx.operands
    return LoweredExpr(
        rust=f"{LINEAR}(&{a}, &{b}, &{c})?",
        helpers=(boundary_helpers(), linear_helper()),
    )


__all__ = ["try_lower"]
