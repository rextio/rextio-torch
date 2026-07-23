"""Fail-closed claims for ``torch.nn.functional.linear`` (Phase A)."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_LINEAR_NO_BIAS,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

LINEAR_TARGET = "torch.nn.functional.linear"
LINEAR_RULE = "rextio-torch/functional-linear-f32-cpu-2d"
LINEAR_NO_BIAS_RULE = "rextio-torch/functional-linear-none-f32-cpu-2d"

_EXPECTED = (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D)
_EXPECTED_NO_BIAS = (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D)


def _has_positional_none(site: ClaimSite) -> bool:
    if len(site.operand_types) != 3 or len(site.operand_literals) != 3:
        return False
    literal = site.operand_literals[2]
    return (
        site.operand_types[2] == "None"
        and literal.is_literal
        and literal.value is None
    )


def _has_keyword_none(site: ClaimSite) -> bool:
    if len(site.keywords) != 1:
        return False
    keyword = site.keywords[0]
    return (
        keyword.name == "bias"
        and keyword.arg_type == "None"
        and keyword.literal.is_literal
        and keyword.literal.value is None
    )


def _has_contradictory_positional_none(site: ClaimSite) -> bool:
    if len(site.operand_types) != 3 or len(site.operand_literals) != 3:
        return False
    literal = site.operand_literals[2]
    return (
        site.operand_types[2] != "None"
        and literal.is_literal
        and literal.value is None
    )


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim the Phase A three-operand functional linear shape, else None."""
    if site.kind != "call" or site.target != LINEAR_TARGET:
        return None
    # Method form is not this lane.
    if site.receiver is not None:
        return NotCovered()
    operands = tuple(site.operand_types)
    positional_none = not site.keywords and _has_positional_none(site)
    keyword_none = len(operands) == 2 and _has_keyword_none(site)
    omitted_none = len(operands) == 2 and not site.keywords
    if positional_none or keyword_none or omitted_none:
        input_weight = operands[:2]
        if any(operand is None for operand in input_weight):
            return NotCovered()
        if any(not is_tensor_type(operand) for operand in input_weight):
            return reject(
                site,
                DIAGNOSTIC_UNSUPPORTED,
                "input/weight types are outside the float32 CPU rank-2 tensor surface",
                "Annotate input and weight as rextio_torch.types.TensorF32Cpu2D.",
            )
        if input_weight != _EXPECTED_NO_BIAS:
            return reject(
                site,
                DIAGNOSTIC_UNSUPPORTED,
                f"linear without bias requires rank-2 input/weight; got {input_weight!r}",
                "Use TensorF32Cpu2D for input and weight.",
            )
        return Claimed(
            rule_id=LINEAR_NO_BIAS_RULE,
            result_type=TENSOR_F32_CPU_2D,
        )

    if _has_contradictory_positional_none(site):
        return reject(
            site,
            DIAGNOSTIC_LINEAR_NO_BIAS,
            "positional bias metadata contradicts literal None",
            "Use a real tensor bias or a literal None whose static type is exactly None.",
        )

    if site.keywords or len(operands) != 3:
        return reject(
            site,
            DIAGNOSTIC_LINEAR_NO_BIAS,
            "linear accepts three positional tensors, or rank-2 input/weight "
            "with omitted/literal-None bias",
            (
                "Pass tensor bias positionally. For no bias, omit it, pass literal "
                "None positionally, or use the exact keyword bias=None."
            ),
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


__all__ = ["LINEAR_NO_BIAS_RULE", "LINEAR_RULE", "LINEAR_TARGET", "try_claim"]
