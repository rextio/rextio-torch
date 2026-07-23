"""Lower exact bounded unary tensor math after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.unary import (
    UNARY_ABS_RULE,
    UNARY_EXP_RULE,
    UNARY_LOG_RULE,
    UNARY_NEGATIVE_RULE,
    UNARY_NEG_RULE,
    UNARY_SQRT_RULE,
    UNARY_SQUARE_RULE,
)
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.rust_snippets import boundary_helpers, unary_call_name, unary_helper

_RULES: dict[str, str] = {
    "abs": UNARY_ABS_RULE,
    "neg": UNARY_NEG_RULE,
    "negative": UNARY_NEGATIVE_RULE,
    "square": UNARY_SQUARE_RULE,
    "exp": UNARY_EXP_RULE,
    "log": UNARY_LOG_RULE,
    "sqrt": UNARY_SQRT_RULE,
}
_FUNCTION_TARGETS: dict[str, str] = {
    f"torch.{operation}": operation for operation in _RULES
}
_METHODS: frozenset[str] = frozenset(_RULES)
_RANK_TYPES: frozenset[str] = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed unary site, or return None for another lane."""
    if claimed.kind != "call":
        return None

    functional_operation = _FUNCTION_TARGETS.get(claimed.target)
    method_operation = _method_name(claimed.target)
    is_functional = functional_operation is not None and claimed.receiver is None
    is_method = (
        claimed.receiver is not None
        and claimed.target not in _FUNCTION_TARGETS
        and method_operation in _METHODS
    )
    if not is_functional and not is_method:
        if functional_operation is not None or method_operation in _METHODS:
            raise ValueError("rextio-torch unary lower received malformed target/receiver metadata")
        return None

    operation = functional_operation if is_functional else method_operation
    if operation is None:
        raise ValueError("rextio-torch unary lower could not resolve the operation")
    expected_rule = _RULES[operation]
    if claimed.rule_id != expected_rule:
        raise ValueError(
            "rextio-torch unary lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {expected_rule!r}"
        )

    if is_functional:
        if (
            claimed.receiver is not None
            or ctx.receiver is not None
            or len(claimed.operand_types) != 1
            or claimed.operand_types[0] not in _RANK_TYPES
            or len(claimed.operand_literals) != 1
            or claimed.operand_literals[0].is_literal
            or claimed.keywords
            or claimed.result_type != claimed.operand_types[0]
            or len(ctx.operands) != 1
        ):
            raise ValueError(
                f"rextio-torch received malformed functional {operation} lower metadata"
            )
        input_name = ctx.operands[0]
    else:
        receiver = claimed.receiver
        if (
            receiver is None
            or receiver.arg_type not in _RANK_TYPES
            or claimed.operand_types
            or claimed.operand_literals
            or claimed.keywords
            or claimed.result_type != receiver.arg_type
            or ctx.receiver is None
            or ctx.operands
        ):
            raise ValueError(
                f"rextio-torch received malformed method {operation} lower metadata"
            )
        input_name = ctx.receiver

    return LoweredExpr(
        rust=f"{unary_call_name(operation)}(&{input_name})?",
        helpers=(boundary_helpers(), unary_helper(operation)),
    )


__all__ = ["try_lower"]
