"""Bounded build-only CUDA E2 claims.

This lane is intentionally narrower than the CPU surface.  It proves only the
canonical ``matmul -> bias add -> relu -> mean(dim=1)`` inference slice for
float32 rank-1/rank-2 tensors that are already resident on ``cuda:0``.
"""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_CUDA_BIAS_ADD,
    DIAGNOSTIC_CUDA_E2,
    DIAGNOSTIC_CUDA_MATMUL,
    DIAGNOSTIC_CUDA_MEAN,
    DIAGNOSTIC_CUDA_RELU,
    TENSOR_F32_CUDA0_1D,
    TENSOR_F32_CUDA0_2D,
    reject,
)

CUDA_MATMUL_RULE = "rextio-torch/cuda0-matmul-f32-2d"
CUDA_BIAS_ADD_RULE = "rextio-torch/cuda0-bias-add-f32-2d-1d"
CUDA_RELU_RULE = "rextio-torch/cuda0-relu-f32-2d"
CUDA_MEAN_DIM1_RULE = "rextio-torch/cuda0-mean-dim1-f32-2d"

CUDA_RULES = frozenset(
    {
        CUDA_MATMUL_RULE,
        CUDA_BIAS_ADD_RULE,
        CUDA_RELU_RULE,
        CUDA_MEAN_DIM1_RULE,
    }
)
CUDA_TYPES = frozenset({TENSOR_F32_CUDA0_1D, TENSOR_F32_CUDA0_2D})


def _has_cuda_type(site: ClaimSite) -> bool:
    if any(item in CUDA_TYPES for item in site.operand_types):
        return True
    return site.receiver is not None and site.receiver.arg_type in CUDA_TYPES


def _literal_keywords(site: ClaimSite) -> dict[str, object] | None:
    values: dict[str, object] = {}
    for keyword in site.keywords:
        if (
            keyword.name in values
            or not keyword.literal.is_literal
            or keyword.name not in {"dim", "keepdim"}
            or (keyword.name == "dim" and keyword.arg_type != "int")
            or (keyword.name == "keepdim" and keyword.arg_type != "bool")
        ):
            return None
        values[keyword.name] = keyword.literal.value
    return values


def _nonliteral_tensor_operands(site: ClaimSite) -> bool:
    return (
        not site.operand_literals
        or (
            len(site.operand_literals) == len(site.operand_types)
            and not any(item.is_literal for item in site.operand_literals)
        )
    )


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim exactly the frozen CUDA E2 operator spellings."""
    if not _has_cuda_type(site):
        return None

    if site.kind == "binop" and site.target == "@":
        if (
            site.receiver is None
            and not site.keywords
            and _nonliteral_tensor_operands(site)
            and site.operand_types
            == (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D)
        ):
            return Claimed(rule_id=CUDA_MATMUL_RULE, result_type=TENSOR_F32_CUDA0_2D)
        return reject(
            site,
            DIAGNOSTIC_CUDA_MATMUL,
            "CUDA matmul is limited to cuda:0 float32 rank-2 @ rank-2",
            "Use TensorF32Cuda0_2D for both @ operands; transfers and mixed devices stay fallback.",
        )

    if site.kind == "binop" and site.target == "+":
        if (
            site.receiver is None
            and not site.keywords
            and _nonliteral_tensor_operands(site)
            and site.operand_types
            == (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_1D)
        ):
            return Claimed(rule_id=CUDA_BIAS_ADD_RULE, result_type=TENSOR_F32_CUDA0_2D)
        return reject(
            site,
            DIAGNOSTIC_CUDA_BIAS_ADD,
            "CUDA add is limited to rank-2 + rank-1 trailing bias broadcast on cuda:0",
            "Keep matrix first and bias second, both float32 and already resident on cuda:0.",
        )

    if site.kind == "call" and site.target.rpartition(".")[2] == "relu":
        if (
            site.receiver is not None
            and site.receiver.arg_type == TENSOR_F32_CUDA0_2D
            and not site.operand_types
            and not site.operand_literals
            and not site.keywords
        ):
            return Claimed(rule_id=CUDA_RELU_RULE, result_type=TENSOR_F32_CUDA0_2D)
        return reject(
            site,
            DIAGNOSTIC_CUDA_RELU,
            "CUDA ReLU is limited to zero-argument rank-2 receiver form",
            "Call .relu() on a TensorF32Cuda0_2D value.",
        )

    if site.kind == "call" and site.target.rpartition(".")[2] == "mean":
        values = _literal_keywords(site)
        if (
            site.receiver is not None
            and site.receiver.arg_type == TENSOR_F32_CUDA0_2D
            and not site.operand_types
            and not site.operand_literals
            and values is not None
            and type(values.get("dim")) is int
            and values.get("dim") == 1
            and type(values.get("keepdim", False)) is bool
            and values.get("keepdim", False) is False
            and "dim" in values
        ):
            return Claimed(
                rule_id=CUDA_MEAN_DIM1_RULE,
                result_type=TENSOR_F32_CUDA0_1D,
            )
        return reject(
            site,
            DIAGNOSTIC_CUDA_MEAN,
            "CUDA mean is limited to literal dim=1 and keepdim=False on rank-2 cuda:0",
            "Call .mean(dim=1) or .mean(dim=1, keepdim=False) on TensorF32Cuda0_2D.",
        )

    # A CUDA tensor reaching any other covered Torch construct must never be
    # admitted by the broader CPU rules.
    if site.kind in {"binop", "call"}:
        return reject(
            site,
            DIAGNOSTIC_CUDA_E2,
            "operation is outside the build-only CUDA E2 vertical slice",
            "Use only @, rank-2 + rank-1 bias, rank-2 .relu(), and rank-2 .mean(dim=1).",
        )
    return NotCovered()


__all__ = [
    "CUDA_BIAS_ADD_RULE",
    "CUDA_MATMUL_RULE",
    "CUDA_MEAN_DIM1_RULE",
    "CUDA_RELU_RULE",
    "CUDA_RULES",
    "CUDA_TYPES",
    "try_claim",
]
