"""Fail-closed claims for tensor ``.relu()`` (Phase A)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_RELU,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

RELU_RULE = "rextio-torch/tensor-relu-f32-cpu-2d"


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim ``.relu()`` on float32 CPU rank-2 receivers, else None."""
    if site.kind != "call" or _method_name(site.target) != "relu":
        return None
    receiver = site.receiver
    if receiver is None:
        # Module-style torch.relu is outside Phase A.
        return NotCovered()
    if site.operand_types or site.keywords:
        return reject(
            site,
            DIAGNOSTIC_RELU,
            "only zero-argument .relu() is supported",
            "Call .relu() with no arguments; do not use relu_ or keyword forms.",
        )
    if receiver.arg_type is None:
        return NotCovered()
    if not is_tensor_type(receiver.arg_type):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "receiver type is outside the float32 CPU tensor surface",
            "Annotate the receiver as rextio_torch.types.TensorF32Cpu2D.",
        )
    if receiver.arg_type != TENSOR_F32_CPU_2D:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f"Phase A ReLU requires float32 CPU rank-2; got {receiver.arg_type!r}",
            "Use TensorF32Cpu2D for the ReLU receiver in the Phase A slice.",
        )
    return Claimed(rule_id=RELU_RULE, result_type=TENSOR_F32_CPU_2D)


__all__ = ["RELU_RULE", "try_claim"]
