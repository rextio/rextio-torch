"""Fail-closed claims for ``.mean`` / ``.sum`` with literal ``dim=1, keepdim=False``."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_MEAN,
    DIAGNOSTIC_SUM,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

MEAN_RULE = "rextio-torch/tensor-mean-dim1-f32-cpu-2d"
SUM_RULE = "rextio-torch/tensor-sum-dim1-f32-cpu-2d"

_REDUCTION_RULES: dict[str, tuple[str, str]] = {
    "mean": (MEAN_RULE, DIAGNOSTIC_MEAN),
    "sum": (SUM_RULE, DIAGNOSTIC_SUM),
}

REDUCTION_RULES: frozenset[str] = frozenset({MEAN_RULE, SUM_RULE})


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
    """Claim mean/sum with literal dim=1, keepdim=False on rank-2, else None."""
    method = _method_name(site.target)
    if site.kind != "call" or method not in _REDUCTION_RULES:
        return None
    rule_id, diagnostic = _REDUCTION_RULES[method]
    receiver = site.receiver
    if receiver is None:
        return NotCovered()
    if site.operand_types:
        return reject(
            site,
            diagnostic,
            f"positional arguments to .{method} are not supported",
            f"Write .{method}(dim=1, keepdim=False) with only literal keywords.",
        )
    keywords = _keyword_map(site)
    if keywords is None or set(keywords) != {"dim", "keepdim"}:
        return reject(
            site,
            diagnostic,
            f"only .{method}(dim=1, keepdim=False) with literal keywords is supported",
            "Pass dim=1 and keepdim=False as static literals; omit other keywords.",
        )
    dim = keywords["dim"]
    keepdim = keywords["keepdim"]
    if not isinstance(dim, int) or isinstance(dim, bool) or dim != 1:
        return reject(
            site,
            diagnostic,
            f"{method} requires dim=1 literal; got dim={dim!r}",
            "Use the literal keyword dim=1.",
        )
    if keepdim is not False:
        return reject(
            site,
            diagnostic,
            f"{method} requires keepdim=False; got keepdim={keepdim!r}",
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
            f"{method} requires float32 CPU rank-2; got {receiver.arg_type!r}",
            "Use TensorF32Cpu2D for the reduction receiver.",
        )
    return Claimed(rule_id=rule_id, result_type=TENSOR_F32_CPU_1D)


__all__ = ["MEAN_RULE", "REDUCTION_RULES", "SUM_RULE", "try_claim"]
