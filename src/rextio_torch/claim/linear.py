"""Fail-closed claims for ``torch.nn.functional.linear`` (Phase A)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_LINEAR,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

LINEAR_TARGET = "torch.nn.functional.linear"
LINEAR_RULE = "rextio-torch/functional-linear-f32-cpu-2d"

_EXPECTED = (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D)


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim the Phase A three-operand functional linear shape, else None."""
    if site.kind != "call" or site.target != LINEAR_TARGET:
        return None
    # Method form is not this lane.
    if site.receiver is not None:
        return NotCovered()
    operands = tuple(site.operand_types)
    if site.keywords or len(operands) != 3:
        return reject(
            site,
            DIAGNOSTIC_LINEAR,
            "only torch.nn.functional.linear(x, weight, bias) with three "
            "positional tensors is supported",
            "Pass input, weight, and bias positionally; omit keywords and optional bias forms.",
        )
    if any(operand is None for operand in operands):
        return NotCovered()
    if any(not is_tensor_type(operand) for operand in operands):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU rank-1/2 tensor surface",
            "Annotate operands with rextio_torch.types.TensorF32Cpu2D / TensorF32Cpu1D.",
        )
    if operands != _EXPECTED:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            (
                "linear requires float32 CPU rank-2 input and weight plus "
                f"rank-1 bias; got {operands!r}"
            ),
            "Use TensorF32Cpu2D for x/weight and TensorF32Cpu1D for bias.",
        )
    return Claimed(rule_id=LINEAR_RULE, result_type=TENSOR_F32_CPU_2D)


__all__ = ["LINEAR_RULE", "LINEAR_TARGET", "try_claim"]
