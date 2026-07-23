"""Fail-closed claims for exact bounded unary tensor math spellings."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_UNARY_ABS,
    DIAGNOSTIC_UNARY_EXP,
    DIAGNOSTIC_UNARY_LOG,
    DIAGNOSTIC_UNARY_NEG,
    DIAGNOSTIC_UNARY_NEGATIVE,
    DIAGNOSTIC_UNARY_SQRT,
    DIAGNOSTIC_UNARY_SQUARE,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    reject,
)

UNARY_ABS_RULE = "rextio-torch/unary-abs-f32-cpu-rank1-2"
UNARY_NEG_RULE = "rextio-torch/unary-neg-f32-cpu-rank1-2"
UNARY_NEGATIVE_RULE = "rextio-torch/unary-negative-f32-cpu-rank1-2"
UNARY_SQUARE_RULE = "rextio-torch/unary-square-f32-cpu-rank1-2"
UNARY_EXP_RULE = "rextio-torch/unary-exp-f32-cpu-rank1-2"
UNARY_LOG_RULE = "rextio-torch/unary-log-f32-cpu-rank1-2"
UNARY_SQRT_RULE = "rextio-torch/unary-sqrt-f32-cpu-rank1-2"

UNARY_RULES: dict[str, str] = {
    "abs": UNARY_ABS_RULE,
    "neg": UNARY_NEG_RULE,
    "negative": UNARY_NEGATIVE_RULE,
    "square": UNARY_SQUARE_RULE,
    "exp": UNARY_EXP_RULE,
    "log": UNARY_LOG_RULE,
    "sqrt": UNARY_SQRT_RULE,
}
UNARY_RULE_SET: frozenset[str] = frozenset(UNARY_RULES.values())

_DIAGNOSTICS: dict[str, str] = {
    "abs": DIAGNOSTIC_UNARY_ABS,
    "neg": DIAGNOSTIC_UNARY_NEG,
    "negative": DIAGNOSTIC_UNARY_NEGATIVE,
    "square": DIAGNOSTIC_UNARY_SQUARE,
    "exp": DIAGNOSTIC_UNARY_EXP,
    "log": DIAGNOSTIC_UNARY_LOG,
    "sqrt": DIAGNOSTIC_UNARY_SQRT,
}
_FUNCTION_TARGETS: dict[str, str] = {
    f"torch.{operation}": operation for operation in UNARY_RULES
}
_METHODS: frozenset[str] = frozenset(UNARY_RULES)
_RANK_TYPES: frozenset[str] = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim exact unary functions and zero-argument tensor methods."""
    if site.kind != "call":
        return None

    if site.receiver is None:
        operation = _FUNCTION_TARGETS.get(site.target)
        if operation is None:
            return None
        diagnostic = _DIAGNOSTICS[operation]
        if site.keywords or len(site.operand_types) != 1:
            return reject(
                site,
                diagnostic,
                f"only {site.target}(tensor) with one positional tensor is supported",
                f"Call {site.target}(tensor) with one positional argument and no keywords.",
            )
        operand_type = site.operand_types[0]
        if operand_type is None:
            return NotCovered()
        if operand_type not in _RANK_TYPES:
            return reject(
                site,
                DIAGNOSTIC_UNSUPPORTED,
                f"{site.target} requires float32 CPU rank-1 or rank-2; got {operand_type!r}",
                "Use TensorF32Cpu1D or TensorF32Cpu2D for the unary operand.",
            )
        return Claimed(rule_id=UNARY_RULES[operation], result_type=operand_type)

    operation = _method_name(site.target)
    if operation not in _METHODS or site.target in _FUNCTION_TARGETS:
        return None
    diagnostic = _DIAGNOSTICS[operation]
    if site.operand_types or site.keywords:
        return reject(
            site,
            diagnostic,
            f"only zero-argument .{operation}() is supported",
            f"Call .{operation}() with no positional arguments or keywords.",
        )
    receiver_type = site.receiver.arg_type
    if receiver_type is None:
        return NotCovered()
    if receiver_type not in _RANK_TYPES:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f".{operation}() requires float32 CPU rank-1 or rank-2; got {receiver_type!r}",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for the unary receiver.",
        )
    return Claimed(rule_id=UNARY_RULES[operation], result_type=receiver_type)


__all__ = [
    "UNARY_ABS_RULE",
    "UNARY_EXP_RULE",
    "UNARY_LOG_RULE",
    "UNARY_NEGATIVE_RULE",
    "UNARY_NEG_RULE",
    "UNARY_RULES",
    "UNARY_RULE_SET",
    "UNARY_SQRT_RULE",
    "UNARY_SQUARE_RULE",
    "try_claim",
]
