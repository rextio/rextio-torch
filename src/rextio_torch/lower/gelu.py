"""Lower exact functional GELU to a fixed ``f_gelu("none")`` helper."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.gelu import GELU_NONE_RULE, GELU_TARGET, has_exact_none_option
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.lower.function_scope import function_scope_guard_active
from rextio_torch.rust_snippets import (
    GELU_NONE,
    boundary_helpers,
    function_scoped_call_name,
    gelu_none_helper,
)

_RANK_TYPES: frozenset[str] = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed exact GELU site, or return None."""
    if claimed.kind != "call":
        return None
    if claimed.target != GELU_TARGET:
        return None
    if claimed.rule_id != GELU_NONE_RULE:
        raise ValueError(
            "rextio-torch GELU lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {GELU_NONE_RULE!r}"
        )
    if (
        claimed.receiver is not None
        or ctx.receiver is not None
        or len(claimed.operand_types) != 1
        or claimed.operand_types[0] not in _RANK_TYPES
        or len(claimed.operand_literals) != 1
        or claimed.operand_literals[0].is_literal
        or claimed.result_type != claimed.operand_types[0]
        or not has_exact_none_option(claimed)
        or len(ctx.operands) != 1
    ):
        raise ValueError("rextio-torch received malformed exact GELU lower metadata")
    scope_active = function_scope_guard_active(ctx)
    call_name = function_scoped_call_name(
        GELU_NONE,
        function_scope_guard_active=scope_active,
    )
    return LoweredExpr(
        rust=f"{call_name}(&{ctx.operands[0]})?",
        helpers=(
            boundary_helpers(),
            gelu_none_helper(function_scope_guard_active=scope_active),
        ),
    )


__all__ = ["try_lower"]
