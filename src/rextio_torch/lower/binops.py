"""Lower elementwise ``+`` and rank-2 matmul claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.binops import (
    ADD_BROADCAST_2D_1D_RULE,
    ADD_RULES,
    ADD_SAME_RANK_RULE,
    MATMUL_BINOP_RULE,
    MATMUL_CALL_RULE,
    MATMUL_CALL_TARGET,
)
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.rust_snippets import (
    ADD,
    MATMUL,
    add_helper,
    boundary_helpers,
    matmul_helper,
)


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


_SAME_RANK_ADD_TYPES: frozenset[tuple[str, str, str]] = frozenset(
    {
        (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
        (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
    }
)
_ADD_F32_TYPES: frozenset[str] = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})
_BROADCAST_ADD_TYPES: frozenset[tuple[str, str, str]] = frozenset(
    {
        (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D),
        (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
    }
)


def _try_lower_add(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    if claimed.kind != "binop" or claimed.target != "+":
        return None
    if claimed.rule_id not in ADD_RULES:
        raise ValueError(
            "rextio-torch add lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    if len(claimed.operand_types) != 2 or claimed.keywords or claimed.receiver is not None:
        raise ValueError("rextio-torch add lower requires two positional tensor operands")
    if len(ctx.operands) != 2:
        raise ValueError(
            "rextio-torch add lower requires two ctx.operands; "
            f"got {len(ctx.operands)}"
        )
    left, right = claimed.operand_types
    if left is None or right is None:
        raise ValueError("rextio-torch add lower requires resolved operand types")
    metadata = (left, right, claimed.result_type)
    if left not in _ADD_F32_TYPES or right not in _ADD_F32_TYPES:
        raise ValueError(
            "rextio-torch add lower requires documented float32 CPU operand types; "
            f"got {left!r}, {right!r}"
        )
    if claimed.rule_id == ADD_SAME_RANK_RULE:
        if metadata not in _SAME_RANK_ADD_TYPES:
            raise ValueError(
                "rextio-torch same-rank add lower metadata changed between claim and lower"
            )
    elif claimed.rule_id == ADD_BROADCAST_2D_1D_RULE:
        if metadata not in _BROADCAST_ADD_TYPES:
            raise ValueError(
                "rextio-torch broadcast add lower metadata changed between claim and lower"
            )
    else:
        raise ValueError(f"rextio-torch add lower unexpected rule_id: {claimed.rule_id!r}")
    a, b = ctx.operands
    return LoweredExpr(
        rust=f"{ADD}(&{a}, &{b})?",
        helpers=(boundary_helpers(), add_helper()),
    )


def _try_lower_matmul(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    is_binop = claimed.kind == "binop" and claimed.target == "@"
    is_functional = claimed.kind == "call" and claimed.target == MATMUL_CALL_TARGET
    is_method = claimed.kind == "call" and _method_name(claimed.target) == "matmul"
    if not is_binop and not is_functional and not is_method:
        return None
    if claimed.keywords or claimed.result_type != TENSOR_F32_CPU_2D:
        raise ValueError("rextio-torch matmul lower requires no keywords and rank-2 result")

    if is_binop:
        if claimed.rule_id != MATMUL_BINOP_RULE:
            raise ValueError(
                "rextio-torch matmul lower received mismatched rule_id: "
                f"{claimed.rule_id!r} != {MATMUL_BINOP_RULE!r}"
            )
        if claimed.receiver is not None or ctx.receiver is not None:
            raise ValueError("rextio-torch binary @ lower cannot carry a receiver")
        if tuple(claimed.operand_types) != (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D):
            raise ValueError(
                "rextio-torch binary @ lower requires two TensorF32Cpu2D operands"
            )
        if len(ctx.operands) != 2:
            raise ValueError(
                "rextio-torch binary @ lower requires two ctx.operands; "
                f"got {len(ctx.operands)}"
            )
        left_name, right_name = ctx.operands
    elif is_functional:
        if claimed.rule_id != MATMUL_CALL_RULE:
            raise ValueError(
                "rextio-torch matmul lower received mismatched rule_id: "
                f"{claimed.rule_id!r} != {MATMUL_CALL_RULE!r}"
            )
        if claimed.receiver is not None or ctx.receiver is not None:
            raise ValueError(
                "rextio-torch functional torch.matmul lower cannot carry a receiver"
            )
        if tuple(claimed.operand_types) != (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D):
            raise ValueError(
                "rextio-torch functional torch.matmul lower requires two "
                "TensorF32Cpu2D operands"
            )
        if len(ctx.operands) != 2:
            raise ValueError(
                "rextio-torch functional torch.matmul lower requires two ctx.operands; "
                f"got {len(ctx.operands)}"
            )
        left_name, right_name = ctx.operands
    elif is_method:
        if claimed.rule_id != MATMUL_CALL_RULE:
            raise ValueError(
                "rextio-torch matmul lower received mismatched rule_id: "
                f"{claimed.rule_id!r} != {MATMUL_CALL_RULE!r}"
            )
        if (
            claimed.receiver is None
            or claimed.receiver.arg_type != TENSOR_F32_CPU_2D
            or len(claimed.operand_types) != 1
            or claimed.operand_types[0] != TENSOR_F32_CPU_2D
        ):
            raise ValueError(
                "rextio-torch method .matmul lower requires a TensorF32Cpu2D "
                "receiver and one TensorF32Cpu2D operand"
            )
        if ctx.receiver is None or len(ctx.operands) != 1:
            raise ValueError(
                "rextio-torch method .matmul lower requires ctx.receiver and one operand"
            )
        left_name = ctx.receiver
        right_name = ctx.operands[0]
    else:
        return None

    return LoweredExpr(
        rust=f"{MATMUL}(&{left_name}, &{right_name})?",
        helpers=(boundary_helpers(), matmul_helper()),
    )


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed + / matmul site, or return None."""
    for lane in (_try_lower_add, _try_lower_matmul):
        result = lane(claimed, ctx)
        if result is not None:
            return result
    return None


__all__ = ["try_lower"]
