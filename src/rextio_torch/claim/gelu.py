"""Fail-closed claim for exact functional GELU with ``approximate='none'``."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_GELU_NONE,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    reject,
)

GELU_TARGET = "torch.nn.functional.gelu"
GELU_NONE_RULE = "rextio-torch/functional-gelu-none-f32-cpu-rank1-2"

_RANK_TYPES: frozenset[str] = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})


def has_exact_none_option(site: ClaimSite) -> bool:
    """Return whether GELU options are omitted or exact literal ``none``."""
    if not site.keywords:
        return True
    if len(site.keywords) != 1:
        return False
    keyword = site.keywords[0]
    return (
        keyword.name == "approximate"
        and keyword.arg_type == "str"
        and keyword.literal.is_literal
        and type(keyword.literal.value) is str
        and keyword.literal.value == "none"
    )


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim exact functional GELU default/explicit-none forms."""
    if site.kind != "call" or site.target != GELU_TARGET:
        return None
    if (
        site.receiver is not None
        or len(site.operand_types) != 1
        or len(site.operand_literals) != 1
        or site.operand_literals[0].is_literal
        or not has_exact_none_option(site)
    ):
        return reject(
            site,
            DIAGNOSTIC_GELU_NONE,
            (
                "GELU requires one positional non-literal tensor and either no "
                "keywords or exact literal approximate='none'"
            ),
            (
                "Call torch.nn.functional.gelu(tensor) or pass only the literal "
                "keyword approximate='none'; omit positional/dynamic/tanh variants."
            ),
        )
    operand_type = site.operand_types[0]
    if operand_type is None:
        return NotCovered()
    if operand_type not in _RANK_TYPES:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f"GELU requires float32 CPU rank-1 or rank-2; got {operand_type!r}",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for the GELU operand.",
        )
    return Claimed(rule_id=GELU_NONE_RULE, result_type=operand_type)


__all__ = ["GELU_NONE_RULE", "GELU_TARGET", "has_exact_none_option", "try_claim"]
