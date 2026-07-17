"""Fail-closed claims for tensor ``.relu()`` / ``.sigmoid()`` / ``.tanh()``.

Supports float32 CPU rank-1 and rank-2 method forms with zero arguments.
Module-style spellings (``torch.relu``) stay outside the Alpha AOT surface.
"""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_RELU,
    DIAGNOSTIC_SIGMOID,
    DIAGNOSTIC_TANH,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

# Rank-2 rule ids keep the Phase A certified constants for existing tests/e2e.
RELU_RULE = "rextio-torch/tensor-relu-f32-cpu-2d"
RELU_RULE_1D = "rextio-torch/tensor-relu-f32-cpu-1d"
SIGMOID_RULE_1D = "rextio-torch/tensor-sigmoid-f32-cpu-1d"
SIGMOID_RULE_2D = "rextio-torch/tensor-sigmoid-f32-cpu-2d"
TANH_RULE_1D = "rextio-torch/tensor-tanh-f32-cpu-1d"
TANH_RULE_2D = "rextio-torch/tensor-tanh-f32-cpu-2d"

_SUPPORTED_RANKS = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})

_RULES: dict[str, dict[str, str]] = {
    "relu": {
        TENSOR_F32_CPU_1D: RELU_RULE_1D,
        TENSOR_F32_CPU_2D: RELU_RULE,
    },
    "sigmoid": {
        TENSOR_F32_CPU_1D: SIGMOID_RULE_1D,
        TENSOR_F32_CPU_2D: SIGMOID_RULE_2D,
    },
    "tanh": {
        TENSOR_F32_CPU_1D: TANH_RULE_1D,
        TENSOR_F32_CPU_2D: TANH_RULE_2D,
    },
}

_DIAGNOSTICS: dict[str, str] = {
    "relu": DIAGNOSTIC_RELU,
    "sigmoid": DIAGNOSTIC_SIGMOID,
    "tanh": DIAGNOSTIC_TANH,
}

ACTIVATION_RULES: frozenset[str] = frozenset(
    rule for by_rank in _RULES.values() for rule in by_rank.values()
)


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim zero-arg activation methods on float32 CPU rank-1/2 receivers."""
    method = _method_name(site.target)
    if site.kind != "call" or method not in _RULES:
        return None
    receiver = site.receiver
    if receiver is None:
        # Module-style torch.relu / torch.sigmoid / torch.tanh stay unclaimed.
        return NotCovered()
    diagnostic = _DIAGNOSTICS[method]
    if site.operand_types or site.keywords:
        return reject(
            site,
            diagnostic,
            f"only zero-argument .{method}() is supported",
            f"Call .{method}() with no arguments; do not use in-place or keyword forms.",
        )
    if receiver.arg_type is None:
        return NotCovered()
    if not is_tensor_type(receiver.arg_type):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "receiver type is outside the float32 CPU tensor surface",
            "Annotate the receiver as rextio_torch.types.TensorF32Cpu1D or TensorF32Cpu2D.",
        )
    if receiver.arg_type not in _SUPPORTED_RANKS:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f".{method}() requires float32 CPU rank-1 or rank-2; got {receiver.arg_type!r}",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for the activation receiver.",
        )
    rule_id = _RULES[method][receiver.arg_type]
    return Claimed(rule_id=rule_id, result_type=receiver.arg_type)


__all__ = [
    "ACTIVATION_RULES",
    "RELU_RULE",
    "RELU_RULE_1D",
    "SIGMOID_RULE_1D",
    "SIGMOID_RULE_2D",
    "TANH_RULE_1D",
    "TANH_RULE_2D",
    "try_claim",
]
