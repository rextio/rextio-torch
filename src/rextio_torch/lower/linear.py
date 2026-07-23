"""Lower bounded functional linear claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.linear import LINEAR_NO_BIAS_RULE, LINEAR_RULE, LINEAR_TARGET
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.rust_snippets import (
    LINEAR,
    LINEAR_NO_BIAS,
    boundary_helpers,
    linear_helper,
    linear_no_bias_helper,
)

_EXPECTED = (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D)
_EXPECTED_NO_BIAS = (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D)


def _keyword_bias_none(claimed: ClaimSite) -> bool:
    if len(claimed.keywords) != 1:
        return False
    keyword = claimed.keywords[0]
    return (
        keyword.name == "bias"
        and keyword.literal.is_literal
        and keyword.literal.value is None
    )


def _positional_bias_none(claimed: ClaimSite) -> bool:
    if len(claimed.operand_types) != 3 or len(claimed.operand_literals) != 3:
        return False
    literal = claimed.operand_literals[2]
    return literal.is_literal and literal.value is None


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed functional linear site, or return ``None``."""
    if claimed.kind != "call" or claimed.target != LINEAR_TARGET:
        return None
    if claimed.receiver is not None or getattr(ctx, "receiver", None) is not None:
        raise ValueError("rextio-torch functional linear lower cannot carry a receiver")
    if claimed.result_type != TENSOR_F32_CPU_2D:
        raise ValueError(
            "rextio-torch linear lower result type changed between claim and lower: "
            f"{claimed.result_type!r}"
        )

    if claimed.rule_id == LINEAR_RULE:
        if claimed.keywords or tuple(claimed.operand_types) != _EXPECTED:
            raise ValueError(
                "rextio-torch linear lower requires three positional tensor operands"
            )
        if len(ctx.operands) != 3:
            raise ValueError(
                "rextio-torch linear lower requires three ctx.operands entries; "
                f"got {len(ctx.operands)}"
            )
        input_name, weight_name, bias_name = ctx.operands
        return LoweredExpr(
            rust=f"{LINEAR}(&{input_name}, &{weight_name}, &{bias_name})?",
            helpers=(boundary_helpers(), linear_helper()),
        )

    if claimed.rule_id != LINEAR_NO_BIAS_RULE:
        raise ValueError(
            "rextio-torch linear lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    if tuple(claimed.operand_types[:2]) != _EXPECTED_NO_BIAS:
        raise ValueError(
            "rextio-torch no-bias linear input/weight changed between claim and lower"
        )

    omitted_none = len(claimed.operand_types) == 2 and not claimed.keywords
    keyword_none = len(claimed.operand_types) == 2 and _keyword_bias_none(claimed)
    positional_none = not claimed.keywords and _positional_bias_none(claimed)
    if not (omitted_none or keyword_none or positional_none):
        raise ValueError(
            "rextio-torch no-bias linear lower requires omitted or literal-None bias"
        )
    if len(ctx.operands) != len(claimed.operand_types):
        raise ValueError(
            "rextio-torch no-bias linear lower operand metadata/rendering misaligned"
        )

    input_name, weight_name = ctx.operands[:2]
    return LoweredExpr(
        # A positional literal None is compile-time option metadata only and is
        # deliberately not forwarded as a runtime tensor operand.
        rust=f"{LINEAR_NO_BIAS}(&{input_name}, &{weight_name})?",
        helpers=(boundary_helpers(), linear_no_bias_helper()),
    )


__all__ = ["try_lower"]
