"""Plugin type keys, diagnostic codes, and rejection helpers."""

from __future__ import annotations

from rextio.analyzer.diagnostics import Diagnostic
from rextio.plugins.api import ClaimSite, Rejected

PLUGIN_ID = "rextio-torch"

TENSOR_F32_CPU_1D = "rextio-torch/tensor-f32-cpu-1d"
TENSOR_F32_CPU_2D = "rextio-torch/tensor-f32-cpu-2d"

TENSOR_TYPE_KEYS: frozenset[str] = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})

# type key -> (dtype token, device token, rank)
_TENSOR_META: dict[str, tuple[str, str, int]] = {
    TENSOR_F32_CPU_1D: ("f32", "cpu", 1),
    TENSOR_F32_CPU_2D: ("f32", "cpu", 2),
}

DIAGNOSTIC_LINEAR = "RXTP-TORCH-001"
DIAGNOSTIC_RELU = "RXTP-TORCH-002"
DIAGNOSTIC_MEAN = "RXTP-TORCH-003"
DIAGNOSTIC_UNSUPPORTED = "RXTP-TORCH-010"

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
    "DIAGNOSTIC_LINEAR",
    "DIAGNOSTIC_MEAN",
    "DIAGNOSTIC_RELU",
    "DIAGNOSTIC_UNSUPPORTED",
    "PLUGIN_ID",
    "RUNTIME_ERRORS",
    "TENSOR_F32_CPU_1D",
    "TENSOR_F32_CPU_2D",
    "TENSOR_TYPE_KEYS",
    "is_tensor_type",
    "reject",
    "tensor_meta",
]
