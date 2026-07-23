"""Materialized torch tensor plugin types for Rextio plugin API 1.6."""

from __future__ import annotations

from rextio.artifacts.models import RuntimeRequirement
from rextio.devices import DeviceValueMetadata
from rextio.plugins.api import BoundaryConversion, PluginType

from rextio_torch.diagnostics import (
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_F32_CUDA0_1D,
    TENSOR_F32_CUDA0_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_torch.rust_snippets.boundary import boundary_helpers


# API 1.3+ collects type-owned module support from accepted function signatures,
# even when a function has no plugin claim and therefore never calls lower().
# Both tensor ranks deliberately own the same exact boundary helper text; core
# deduplicates it against the other rank and against claim-level helpers.
_BOUNDARY_SUPPORT = boundary_helpers()


CUDA_RUNTIME_REQUIREMENTS = (
    RuntimeRequirement("libtorch", "2.11.0", ("cuda", "pytorch-wheel")),
    RuntimeRequirement("tch", "0.24.0", ("cuda",)),
)


def _tensor_type(
    key: str,
    annotation: str,
    extractor: str,
    *,
    device_value_metadata: DeviceValueMetadata | None = None,
) -> PluginType:
    materializer = "__rxttorch_materialize_tensor"
    if device_value_metadata is not None:
        materializer = f"__rxttorch_materialize_f32_cuda0_{device_value_metadata.rank}d"
    return PluginType(
        key=key,
        annotations=(f"rextio_torch.types.{annotation}",),
        rust_type="RxtTorchTensor",
        conversion=BoundaryConversion(
            param_rust="pyo3::Bound<'py, pyo3::types::PyAny>",
            param_expr=f"{extractor}(py, &{{param}})?",
            return_rust="pyo3::Bound<'py, pyo3::types::PyAny>",
            return_expr=f"{materializer}(py, {{value}})?",
        ),
        helpers=(_BOUNDARY_SUPPORT,),
        device_value_metadata=device_value_metadata,
    )


def _cuda_metadata(rank: int) -> DeviceValueMetadata:
    return DeviceValueMetadata(
        logical_device="cuda:0",
        backend="cuda",
        dtype="float32",
        rank=rank,
        layout="strided",
        runtime="libtorch",
        runtime_version="2.11.0",
        reuse_domain_runtime=True,
        features=("inference", "no-grad"),
        memory_spaces=("device",),
        runtime_requirements=CUDA_RUNTIME_REQUIREMENTS,
    )


PLUGIN_TYPES: tuple[PluginType, ...] = (
    _tensor_type(
        TENSOR_F32_CPU_2D,
        "TensorF32Cpu2D",
        "__rxttorch_extract_f32_cpu_2d",
    ),
    _tensor_type(
        TENSOR_F32_CPU_1D,
        "TensorF32Cpu1D",
        "__rxttorch_extract_f32_cpu_1d",
    ),
    _tensor_type(
        TENSOR_I64_CPU_1D,
        "TensorI64Cpu1D",
        "__rxttorch_extract_i64_cpu_1d",
    ),
    _tensor_type(
        TENSOR_F32_CUDA0_2D,
        "TensorF32Cuda0_2D",
        "__rxttorch_extract_f32_cuda0_2d",
        device_value_metadata=_cuda_metadata(2),
    ),
    _tensor_type(
        TENSOR_F32_CUDA0_1D,
        "TensorF32Cuda0_1D",
        "__rxttorch_extract_f32_cuda0_1d",
        device_value_metadata=_cuda_metadata(1),
    ),
)

_BY_KEY = {plugin_type.key: plugin_type for plugin_type in PLUGIN_TYPES}


def plugin_types() -> tuple[PluginType, ...]:
    """Return the Phase A materialized tensor vocabulary."""
    return PLUGIN_TYPES


def plugin_type(key: str) -> PluginType:
    """Return one registered type by key."""
    return _BY_KEY[key]


def plugin_type_keys() -> frozenset[str]:
    """Return all owned type keys."""
    return frozenset(_BY_KEY)


__all__ = [
    "CUDA_RUNTIME_REQUIREMENTS",
    "PLUGIN_TYPES",
    "plugin_type",
    "plugin_type_keys",
    "plugin_types",
]
