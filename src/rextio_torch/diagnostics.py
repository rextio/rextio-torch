"""Plugin type keys, diagnostic codes, and rejection helpers."""

from __future__ import annotations

from rextio.analyzer.diagnostics import Diagnostic
from rextio.plugins.api import ClaimSite, Rejected

PLUGIN_ID = "rextio-torch"

TENSOR_F32_CPU_1D = "rextio-torch/tensor-f32-cpu-1d"
TENSOR_F32_CPU_2D = "rextio-torch/tensor-f32-cpu-2d"
TENSOR_I64_CPU_1D = "rextio-torch/tensor-i64-cpu-1d"

TENSOR_TYPE_KEYS: frozenset[str] = frozenset(
    {TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D, TENSOR_I64_CPU_1D}
)

# type key -> (dtype token, device token, rank)
_TENSOR_META: dict[str, tuple[str, str, int]] = {
    TENSOR_F32_CPU_1D: ("f32", "cpu", 1),
    TENSOR_F32_CPU_2D: ("f32", "cpu", 2),
    TENSOR_I64_CPU_1D: ("i64", "cpu", 1),
}

# Per-rule diagnostic codes (must be unique across RuleRecord entries).
DIAGNOSTIC_LINEAR = "RXTP-TORCH-001"
DIAGNOSTIC_RELU = "RXTP-TORCH-002"  # rank-2 ReLU (Phase A certified)
DIAGNOSTIC_MEAN = "RXTP-TORCH-003"
DIAGNOSTIC_RELU_1D = "RXTP-TORCH-004"
DIAGNOSTIC_SIGMOID_1D = "RXTP-TORCH-005"
DIAGNOSTIC_SIGMOID_2D = "RXTP-TORCH-006"
DIAGNOSTIC_TANH_1D = "RXTP-TORCH-007"
DIAGNOSTIC_TANH_2D = "RXTP-TORCH-008"
DIAGNOSTIC_MATMUL = "RXTP-TORCH-009"  # binary @ form
DIAGNOSTIC_UNSUPPORTED = "RXTP-TORCH-010"
DIAGNOSTIC_ADD_SAME_RANK = "RXTP-TORCH-011"
DIAGNOSTIC_ADD_BROADCAST = "RXTP-TORCH-012"
DIAGNOSTIC_SUM = "RXTP-TORCH-013"
DIAGNOSTIC_MATMUL_CALL = "RXTP-TORCH-014"
DIAGNOSTIC_SOFTMAX = "RXTP-TORCH-015"
DIAGNOSTIC_ARGMAX = "RXTP-TORCH-016"
DIAGNOSTIC_MUL_SAME_RANK = "RXTP-TORCH-017"
DIAGNOSTIC_MUL_BROADCAST = "RXTP-TORCH-018"
DIAGNOSTIC_SUB_SAME_RANK = "RXTP-TORCH-019"
DIAGNOSTIC_SUB_BROADCAST = "RXTP-TORCH-020"
DIAGNOSTIC_DIV_SAME_RANK = "RXTP-TORCH-021"
DIAGNOSTIC_DIV_BROADCAST = "RXTP-TORCH-022"

# Shared claim-time family codes for activation/binop shape rejections.
DIAGNOSTIC_SIGMOID = DIAGNOSTIC_SIGMOID_2D
DIAGNOSTIC_TANH = DIAGNOSTIC_TANH_2D
DIAGNOSTIC_ADD = DIAGNOSTIC_ADD_SAME_RANK
DIAGNOSTIC_MUL = DIAGNOSTIC_MUL_SAME_RANK
DIAGNOSTIC_SUB = DIAGNOSTIC_SUB_SAME_RANK
DIAGNOSTIC_DIV = DIAGNOSTIC_DIV_SAME_RANK

RUNTIME_ERRORS = {
    "not_tensor": "rextio-torch: expected a torch.Tensor",
    "device": "rextio-torch: expected a CPU tensor",
    "dtype": "rextio-torch: expected a float32 tensor",
    "rank": "rextio-torch: tensor rank does not match the annotated type",
}


def tensor_meta(type_key: str) -> tuple[str, str, int] | None:
    """Return ``(dtype, device, rank)`` for a plugin tensor key, else None."""
    return _TENSOR_META.get(type_key)


def is_tensor_type(type_key: str | None) -> bool:
    """Report whether ``type_key`` is one of this plugin's tensor types."""
    return type_key is not None and type_key in TENSOR_TYPE_KEYS


def reject(site: ClaimSite, code: str, message: str, suggestion: str) -> Rejected:
    """Build a location-neutral plugin rejection; core stamps the source site."""
    return Rejected(
        diagnostic=Diagnostic(
            code=code,
            severity="error",
            message=f"rextio-torch cannot lower {site.target!r}: {message}",
            file_path="",
            line=0,
            column=0,
            suggestion=suggestion,
        )
    )


__all__ = [
    "DIAGNOSTIC_ADD",
    "DIAGNOSTIC_ADD_BROADCAST",
    "DIAGNOSTIC_ADD_SAME_RANK",
    "DIAGNOSTIC_LINEAR",
    "DIAGNOSTIC_MATMUL",
    "DIAGNOSTIC_MATMUL_CALL",
    "DIAGNOSTIC_MEAN",
    "DIAGNOSTIC_DIV",
    "DIAGNOSTIC_DIV_BROADCAST",
    "DIAGNOSTIC_DIV_SAME_RANK",
    "DIAGNOSTIC_MUL",
    "DIAGNOSTIC_MUL_BROADCAST",
    "DIAGNOSTIC_MUL_SAME_RANK",
    "DIAGNOSTIC_RELU",
    "DIAGNOSTIC_RELU_1D",
    "DIAGNOSTIC_SIGMOID",
    "DIAGNOSTIC_SIGMOID_1D",
    "DIAGNOSTIC_SIGMOID_2D",
    "DIAGNOSTIC_SOFTMAX",
    "DIAGNOSTIC_SUM",
    "DIAGNOSTIC_SUB",
    "DIAGNOSTIC_SUB_BROADCAST",
    "DIAGNOSTIC_SUB_SAME_RANK",
    "DIAGNOSTIC_TANH",
    "DIAGNOSTIC_TANH_1D",
    "DIAGNOSTIC_TANH_2D",
    "DIAGNOSTIC_UNSUPPORTED",
    "DIAGNOSTIC_ARGMAX",
    "PLUGIN_ID",
    "RUNTIME_ERRORS",
    "TENSOR_F32_CPU_1D",
    "TENSOR_F32_CPU_2D",
    "TENSOR_I64_CPU_1D",
    "TENSOR_TYPE_KEYS",
    "is_tensor_type",
    "reject",
    "tensor_meta",
]
