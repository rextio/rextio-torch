"""Fail-closed claims for tensor ``.mean(dim=1, keepdim=False)`` (Phase A)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_MEAN,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

MEAN_RULE = "rextio-torch/tensor-mean-dim1-f32-cpu-2d"


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def _keyword_map(site: ClaimSite) -> dict[str, object] | None:
    """Return name→literal value for fully static keywords, else None."""
    values: dict[str, object] = {}
    for keyword in site.keywords:
        if not keyword.literal.is_literal:
            return None
        values[keyword.name] = keyword.literal.value
    return values


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim the Phase A mean reduction, else None."""
    if site.kind != "call" or _method_name(site.target) != "mean":
        return None
    receiver = site.receiver
    if receiver is None:
        return NotCovered()
    if site.operand_types:
        return reject(
            site,
            DIAGNOSTIC_MEAN,
            "positional arguments to .mean are not supported in Phase A",
            "Write .mean(dim=1, keepdim=False) with only literal keywords.",
        )
    keywords = _keyword_map(site)
    if keywords is None or set(keywords) != {"dim", "keepdim"}:
        return reject(
            site,
            DIAGNOSTIC_MEAN,
            "only .mean(dim=1, keepdim=False) with literal keywords is supported",
            "Pass dim=1 and keepdim=False as static literals; omit other keywords.",
        )
    dim = keywords["dim"]
    keepdim = keywords["keepdim"]
    if not isinstance(dim, int) or isinstance(dim, bool) or dim != 1:
        return reject(
            site,
            DIAGNOSTIC_MEAN,
            f"Phase A mean requires dim=1 literal; got dim={dim!r}",
            "Use the literal keyword dim=1.",
        )
    if keepdim is not False:
        return reject(
            site,
            DIAGNOSTIC_MEAN,
            f"Phase A mean requires keepdim=False; got keepdim={keepdim!r}",
            "Use the literal keyword keepdim=False.",
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
            f"Phase A mean requires float32 CPU rank-2; got {receiver.arg_type!r}",
            "Use TensorF32Cpu2D for the mean receiver in the Phase A slice.",
        )
    return Claimed(rule_id=MEAN_RULE, result_type=TENSOR_F32_CPU_1D)


__all__ = ["MEAN_RULE", "try_claim"]
