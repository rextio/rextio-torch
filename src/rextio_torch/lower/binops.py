"""Lower elementwise binary ops and rank-2 matmul claims after revalidation."""

from __future__ import annotations

from collections.abc import Callable

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.binops import (
    ADD_BROADCAST_2D_1D_RULE,
    ADD_RULES,
    ADD_SAME_RANK_RULE,
    DIV_BROADCAST_2D_1D_RULE,
    DIV_RULES,
    DIV_SAME_RANK_RULE,
    FUNCTION_ADD_RULE,
    FUNCTION_DIV_RULE,
    FUNCTION_MUL_RULE,
    FUNCTION_SUB_RULE,
    MATMUL_BINOP_MIXED_RANK_RULE,
    MATMUL_BINOP_RULE,
    MATMUL_CALL_MIXED_RANK_RULE,
    MATMUL_CALL_RULE,
    MATMUL_CALL_TARGET,
    MUL_BROADCAST_2D_1D_RULE,
    MUL_RULES,
    MUL_SAME_RANK_RULE,
    SUB_BROADCAST_2D_1D_RULE,
    SUB_RULES,
    SUB_SAME_RANK_RULE,
)
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.lower.function_scope import function_scope_guard_active
from rextio_torch.rust_snippets import (
    ADD,
    DIV,
    MATMUL,
    MUL,
    SUB,
    add_helper,
    boundary_helpers,
    div_helper,
    function_scoped_call_name,
    matmul_helper,
    mul_helper,
    sub_helper,
)


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


_SAME_RANK_TENSOR_TYPES: frozenset[tuple[str, str, str]] = frozenset(
    {
        (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
        (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
    }
)
_F32_TENSOR_TYPES: frozenset[str] = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})
_BROADCAST_TENSOR_TYPES: frozenset[tuple[str, str, str]] = frozenset(
    {
        (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D),
        (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
    }
)
_FUNCTION_ELEMENTWISE = {
    "torch.add": (FUNCTION_ADD_RULE, ADD, add_helper),
    "torch.sub": (FUNCTION_SUB_RULE, SUB, sub_helper),
    "torch.mul": (FUNCTION_MUL_RULE, MUL, mul_helper),
    "torch.div": (FUNCTION_DIV_RULE, DIV, div_helper),
}

_MATMUL_RESULT_TYPES: dict[tuple[str, str], str] = {
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_1D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_1D,
}


def _scope_variant(
    ctx: LoweringContext,
    base_call_name: str,
    helper_factory: Callable[..., str],
) -> tuple[str, str]:
    scope_active = function_scope_guard_active(ctx)
    return (
        function_scoped_call_name(
            base_call_name,
            function_scope_guard_active=scope_active,
        ),
        helper_factory(function_scope_guard_active=scope_active),
    )


def _try_lower_functional_elementwise(
    claimed: ClaimSite,
    ctx: LoweringContext,
) -> LoweredExpr | None:
    if claimed.kind != "call" or claimed.target not in _FUNCTION_ELEMENTWISE:
        return None
    expected_rule, base_call_name, helper_factory = _FUNCTION_ELEMENTWISE[claimed.target]
    if claimed.rule_id != expected_rule:
        raise ValueError(
            "rextio-torch functional elementwise lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {expected_rule!r}"
        )
    if (
        claimed.receiver is not None
        or ctx.receiver is not None
        or claimed.keywords
        or len(claimed.operand_types) != 2
        or len(claimed.operand_literals) != 2
        or any(literal.is_literal for literal in claimed.operand_literals)
        or len(ctx.operands) != 2
    ):
        raise ValueError(
            "rextio-torch functional elementwise lower requires two positional "
            "non-literal tensor operands"
        )
    left, right = claimed.operand_types
    metadata = (left, right, claimed.result_type)
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        raise ValueError(
            "rextio-torch functional elementwise lower requires float32 CPU "
            f"rank-1/2 operands; got {left!r}, {right!r}"
        )
    if metadata not in _SAME_RANK_TENSOR_TYPES | _BROADCAST_TENSOR_TYPES:
        raise ValueError(
            "rextio-torch functional elementwise result metadata changed between "
            "claim and lower"
        )
    left_name, right_name = ctx.operands
    call_name, helper = _scope_variant(ctx, base_call_name, helper_factory)
    return LoweredExpr(
        rust=f"{call_name}(&{left_name}, &{right_name})?",
        helpers=(boundary_helpers(), helper),
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
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        raise ValueError(
            "rextio-torch add lower requires documented float32 CPU operand types; "
            f"got {left!r}, {right!r}"
        )
    if claimed.rule_id == ADD_SAME_RANK_RULE:
        if metadata not in _SAME_RANK_TENSOR_TYPES:
            raise ValueError(
                "rextio-torch same-rank add lower metadata changed between claim and lower"
            )
    elif claimed.rule_id == ADD_BROADCAST_2D_1D_RULE:
        if metadata not in _BROADCAST_TENSOR_TYPES:
            raise ValueError(
                "rextio-torch broadcast add lower metadata changed between claim and lower"
            )
    else:
        raise ValueError(f"rextio-torch add lower unexpected rule_id: {claimed.rule_id!r}")
    a, b = ctx.operands
    call_name, helper = _scope_variant(ctx, ADD, add_helper)
    return LoweredExpr(
        rust=f"{call_name}(&{a}, &{b})?",
        helpers=(boundary_helpers(), helper),
    )


def _try_lower_mul(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    if claimed.kind != "binop" or claimed.target != "*":
        return None
    if claimed.rule_id not in MUL_RULES:
        raise ValueError(
            "rextio-torch multiply lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    if (
        len(claimed.operand_types) != 2
        or claimed.keywords
        or claimed.receiver is not None
        or ctx.receiver is not None
    ):
        raise ValueError("rextio-torch multiply lower requires two positional tensor operands")
    if len(ctx.operands) != 2:
        raise ValueError(
            "rextio-torch multiply lower requires two ctx.operands; "
            f"got {len(ctx.operands)}"
        )
    left, right = claimed.operand_types
    if left is None or right is None:
        raise ValueError("rextio-torch multiply lower requires resolved operand types")
    metadata = (left, right, claimed.result_type)
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        raise ValueError(
            "rextio-torch multiply lower requires documented float32 CPU operand types; "
            f"got {left!r}, {right!r}"
        )
    if claimed.rule_id == MUL_SAME_RANK_RULE:
        if metadata not in _SAME_RANK_TENSOR_TYPES:
            raise ValueError(
                "rextio-torch same-rank multiply lower metadata changed between claim and lower"
            )
    elif claimed.rule_id == MUL_BROADCAST_2D_1D_RULE:
        if metadata not in _BROADCAST_TENSOR_TYPES:
            raise ValueError(
                "rextio-torch broadcast multiply lower metadata changed between claim and lower"
            )
    else:
        raise ValueError(
            f"rextio-torch multiply lower unexpected rule_id: {claimed.rule_id!r}"
        )
    a, b = ctx.operands
    call_name, helper = _scope_variant(ctx, MUL, mul_helper)
    return LoweredExpr(
        rust=f"{call_name}(&{a}, &{b})?",
        helpers=(boundary_helpers(), helper),
    )


def _try_lower_sub(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    if claimed.kind != "binop" or claimed.target != "-":
        return None
    if claimed.rule_id not in SUB_RULES:
        raise ValueError(
            "rextio-torch subtraction lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    if (
        len(claimed.operand_types) != 2
        or claimed.keywords
        or claimed.receiver is not None
        or ctx.receiver is not None
    ):
        raise ValueError("rextio-torch subtraction lower requires two positional tensor operands")
    if len(ctx.operands) != 2:
        raise ValueError(
            "rextio-torch subtraction lower requires two ctx.operands; "
            f"got {len(ctx.operands)}"
        )
    left, right = claimed.operand_types
    if left is None or right is None:
        raise ValueError("rextio-torch subtraction lower requires resolved operand types")
    metadata = (left, right, claimed.result_type)
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        raise ValueError(
            "rextio-torch subtraction lower requires documented float32 CPU operand types; "
            f"got {left!r}, {right!r}"
        )
    if claimed.rule_id == SUB_SAME_RANK_RULE:
        if metadata not in _SAME_RANK_TENSOR_TYPES:
            raise ValueError(
                "rextio-torch same-rank subtraction metadata changed between claim and lower"
            )
    elif claimed.rule_id == SUB_BROADCAST_2D_1D_RULE:
        if metadata not in _BROADCAST_TENSOR_TYPES:
            raise ValueError(
                "rextio-torch broadcast subtraction metadata changed between claim and lower"
            )
    else:
        raise ValueError(
            f"rextio-torch subtraction lower unexpected rule_id: {claimed.rule_id!r}"
        )
    left_name, right_name = ctx.operands
    call_name, helper = _scope_variant(ctx, SUB, sub_helper)
    return LoweredExpr(
        rust=f"{call_name}(&{left_name}, &{right_name})?",
        helpers=(boundary_helpers(), helper),
    )


def _try_lower_div(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    if claimed.kind != "binop" or claimed.target != "/":
        return None
    if claimed.rule_id not in DIV_RULES:
        raise ValueError(
            "rextio-torch division lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    if (
        len(claimed.operand_types) != 2
        or claimed.keywords
        or claimed.receiver is not None
        or ctx.receiver is not None
    ):
        raise ValueError("rextio-torch division lower requires two positional tensor operands")
    if len(ctx.operands) != 2:
        raise ValueError(
            "rextio-torch division lower requires two ctx.operands; "
            f"got {len(ctx.operands)}"
        )
    left, right = claimed.operand_types
    if left is None or right is None:
        raise ValueError("rextio-torch division lower requires resolved operand types")
    metadata = (left, right, claimed.result_type)
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        raise ValueError(
            "rextio-torch division lower requires documented float32 CPU operand types; "
            f"got {left!r}, {right!r}"
        )
    if claimed.rule_id == DIV_SAME_RANK_RULE:
        if metadata not in _SAME_RANK_TENSOR_TYPES:
            raise ValueError(
                "rextio-torch same-rank division metadata changed between claim and lower"
            )
    elif claimed.rule_id == DIV_BROADCAST_2D_1D_RULE:
        if metadata not in _BROADCAST_TENSOR_TYPES:
            raise ValueError(
                "rextio-torch broadcast division metadata changed between claim and lower"
            )
    else:
        raise ValueError(
            f"rextio-torch division lower unexpected rule_id: {claimed.rule_id!r}"
        )
    left_name, right_name = ctx.operands
    call_name, helper = _scope_variant(ctx, DIV, div_helper)
    return LoweredExpr(
        rust=f"{call_name}(&{left_name}, &{right_name})?",
        helpers=(boundary_helpers(), helper),
    )


def _try_lower_matmul(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    is_binop = claimed.kind == "binop" and claimed.target == "@"
    is_functional = claimed.kind == "call" and claimed.target == MATMUL_CALL_TARGET
    is_method = claimed.kind == "call" and _method_name(claimed.target) == "matmul"
    if not is_binop and not is_functional and not is_method:
        return None
    if claimed.keywords:
        raise ValueError("rextio-torch matmul lower requires no keywords")

    if is_binop:
        if claimed.receiver is not None or ctx.receiver is not None:
            raise ValueError("rextio-torch binary @ lower cannot carry a receiver")
        if len(claimed.operand_types) != 2:
            raise ValueError("rextio-torch binary @ lower requires two tensor operands")
        left_type, right_type = claimed.operand_types
        result_type = _MATMUL_RESULT_TYPES.get((left_type, right_type))
        expected_rule = (
            MATMUL_BINOP_RULE
            if (left_type, right_type) == (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D)
            else MATMUL_BINOP_MIXED_RANK_RULE
        )
        if claimed.rule_id != expected_rule:
            raise ValueError(
                "rextio-torch matmul lower received mismatched rule_id: "
                f"{claimed.rule_id!r} != {expected_rule!r}"
            )
        if result_type is None or claimed.result_type != result_type:
            raise ValueError(
                "rextio-torch binary @ lower received unsupported operand/result metadata"
            )
        if len(ctx.operands) != 2:
            raise ValueError(
                "rextio-torch binary @ lower requires two ctx.operands; "
                f"got {len(ctx.operands)}"
            )
        left_name, right_name = ctx.operands
    elif is_functional:
        if claimed.receiver is not None or ctx.receiver is not None:
            raise ValueError(
                "rextio-torch functional torch.matmul lower cannot carry a receiver"
            )
        if len(claimed.operand_types) != 2:
            raise ValueError(
                "rextio-torch functional torch.matmul lower requires two tensor operands"
            )
        left_type, right_type = claimed.operand_types
        result_type = _MATMUL_RESULT_TYPES.get((left_type, right_type))
        expected_rule = (
            MATMUL_CALL_RULE
            if (left_type, right_type) == (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D)
            else MATMUL_CALL_MIXED_RANK_RULE
        )
        if claimed.rule_id != expected_rule:
            raise ValueError(
                "rextio-torch matmul lower received mismatched rule_id: "
                f"{claimed.rule_id!r} != {expected_rule!r}"
            )
        if result_type is None or claimed.result_type != result_type:
            raise ValueError(
                "rextio-torch functional torch.matmul lower received unsupported "
                "operand/result metadata"
            )
        if len(ctx.operands) != 2:
            raise ValueError(
                "rextio-torch functional torch.matmul lower requires two ctx.operands; "
                f"got {len(ctx.operands)}"
            )
        left_name, right_name = ctx.operands
    elif is_method:
        if (
            claimed.receiver is None
            or len(claimed.operand_types) != 1
        ):
            raise ValueError(
                "rextio-torch method .matmul lower requires a typed receiver "
                "and one tensor operand"
            )
        left_type = claimed.receiver.arg_type
        right_type = claimed.operand_types[0]
        result_type = _MATMUL_RESULT_TYPES.get((left_type, right_type))
        expected_rule = (
            MATMUL_CALL_RULE
            if (left_type, right_type) == (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D)
            else MATMUL_CALL_MIXED_RANK_RULE
        )
        if claimed.rule_id != expected_rule:
            raise ValueError(
                "rextio-torch matmul lower received mismatched rule_id: "
                f"{claimed.rule_id!r} != {expected_rule!r}"
            )
        if result_type is None or claimed.result_type != result_type:
            raise ValueError(
                "rextio-torch method .matmul lower received unsupported "
                "receiver/operand/result metadata"
            )
        if ctx.receiver is None or len(ctx.operands) != 1:
            raise ValueError(
                "rextio-torch method .matmul lower requires ctx.receiver and one operand"
            )
        left_name = ctx.receiver
        right_name = ctx.operands[0]
    else:
        return None

    call_name, helper = _scope_variant(ctx, MATMUL, matmul_helper)
    return LoweredExpr(
        rust=f"{call_name}(&{left_name}, &{right_name})?",
        helpers=(boundary_helpers(), helper),
    )


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed + / matmul site, or return None."""
    for lane in (
        _try_lower_functional_elementwise,
        _try_lower_add,
        _try_lower_mul,
        _try_lower_sub,
        _try_lower_div,
        _try_lower_matmul,
    ):
        result = lane(claimed, ctx)
        if result is not None:
            return result
    return None


__all__ = ["try_lower"]
