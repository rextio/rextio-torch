"""Fail-closed claims for tensor ``.relu()`` / ``.sigmoid()`` / ``.tanh()``.

Supports float32 CPU rank-1 and rank-2 method forms with zero arguments.
The exact functional spellings ``torch.relu`` / ``torch.sigmoid`` /
``torch.tanh`` accept one positional tensor and no keywords. The deliberately
separate ``torch.nn.functional.relu`` alias permits only its proved
``inplace=False`` literal option.
"""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_RELU,
    DIAGNOSTIC_FUNCTION_RELU,
    DIAGNOSTIC_FUNCTION_SIGMOID,
    DIAGNOSTIC_FUNCTION_TANH,
    DIAGNOSTIC_FUNCTIONAL_RELU,
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
FUNCTION_RELU_RULE = "rextio-torch/function-relu-f32-cpu-rank1-2"
FUNCTION_SIGMOID_RULE = "rextio-torch/function-sigmoid-f32-cpu-rank1-2"
FUNCTION_TANH_RULE = "rextio-torch/function-tanh-f32-cpu-rank1-2"
FUNCTIONAL_RELU_RULE = "rextio-torch/functional-relu-f32-cpu-rank1-2"

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

_FUNCTION_TARGETS: dict[str, str] = {
    "torch.relu": "relu",
    "torch.sigmoid": "sigmoid",
    "torch.tanh": "tanh",
    "torch.nn.functional.relu": "relu",
}

_FUNCTION_RULES: dict[str, str] = {
    "relu": FUNCTION_RELU_RULE,
    "sigmoid": FUNCTION_SIGMOID_RULE,
    "tanh": FUNCTION_TANH_RULE,
}

_FUNCTION_DIAGNOSTICS: dict[str, str] = {
    "relu": DIAGNOSTIC_FUNCTION_RELU,
    "sigmoid": DIAGNOSTIC_FUNCTION_SIGMOID,
    "tanh": DIAGNOSTIC_FUNCTION_TANH,
}

ACTIVATION_RULES: frozenset[str] = frozenset(
    {
        *(rule for by_rank in _RULES.values() for rule in by_rank.values()),
        *_FUNCTION_RULES.values(),
        FUNCTIONAL_RELU_RULE,
    }
)


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim bounded method and exact functional activation spellings."""
    method = _method_name(site.target)
    if site.kind != "call" or method not in _RULES:
        return None
    receiver = site.receiver
    if receiver is None:
        functional_method = _FUNCTION_TARGETS.get(site.target)
        if functional_method is None:
            return NotCovered()
        functional_relu_alias = site.target == "torch.nn.functional.relu"
        diagnostic = (
            DIAGNOSTIC_FUNCTIONAL_RELU
            if functional_relu_alias
            else _FUNCTION_DIAGNOSTICS[functional_method]
        )
        if len(site.operand_types) != 1:
            return reject(
                site,
                diagnostic,
                f"only {site.target}(tensor) with one positional tensor is supported",
                f"Call {site.target}(tensor) with one positional tensor.",
            )
        if functional_relu_alias:
            if len(site.operand_literals) != 1:
                return reject(
                    site,
                    diagnostic,
                    "functional ReLU literal metadata must align with its tensor input",
                    "Use one tensor input and omit inplace or write inplace=False.",
                )
            if len(site.keywords) > 1:
                return reject(
                    site,
                    diagnostic,
                    "functional ReLU accepts at most one inplace=False literal",
                    "Omit inplace or use the exact literal inplace=False.",
                )
            if site.keywords:
                keyword = site.keywords[0]
                if not (
                    keyword.name == "inplace"
                    and keyword.arg_type == "bool"
                    and keyword.literal.is_literal
                    and keyword.literal.value is False
                ):
                    return reject(
                        site,
                        diagnostic,
                        "functional ReLU accepts only the exact literal inplace=False option",
                        "Omit inplace or use the exact literal inplace=False.",
                    )
        elif site.keywords:
            return reject(
                site,
                diagnostic,
                f"only {site.target}(tensor) with one positional tensor is supported",
                f"Call {site.target}(tensor) with one positional argument and no keywords.",
            )
        operand_type = site.operand_types[0]
        if operand_type is None:
            return NotCovered()
        if operand_type not in _SUPPORTED_RANKS:
            return reject(
                site,
                DIAGNOSTIC_UNSUPPORTED,
                f"{site.target} requires float32 CPU rank-1 or rank-2; got {operand_type!r}",
                "Use TensorF32Cpu1D or TensorF32Cpu2D for the activation operand.",
            )
        return Claimed(
            rule_id=(
                FUNCTIONAL_RELU_RULE
                if functional_relu_alias
                else _FUNCTION_RULES[functional_method]
            ),
            result_type=operand_type,
        )
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
    "FUNCTION_RELU_RULE",
    "FUNCTIONAL_RELU_RULE",
    "FUNCTION_SIGMOID_RULE",
    "FUNCTION_TANH_RULE",
    "RELU_RULE",
    "RELU_RULE_1D",
    "SIGMOID_RULE_1D",
    "SIGMOID_RULE_2D",
    "TANH_RULE_1D",
    "TANH_RULE_2D",
    "try_claim",
]
