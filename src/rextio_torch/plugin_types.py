"""Materialized torch tensor plugin types for Rextio plugin API 1.3."""

from __future__ import annotations

from rextio.plugins.api import BoundaryConversion, PluginType

from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D, TENSOR_I64_CPU_1D
from rextio_torch.rust_snippets.boundary import boundary_helpers


# API 1.3 collects type-owned module support from accepted function signatures,
# even when a function has no plugin claim and therefore never calls lower().
# Both tensor ranks deliberately own the same exact boundary helper text; core
# deduplicates it against the other rank and against claim-level helpers.
_BOUNDARY_SUPPORT = boundary_helpers()


def _tensor_type(key: str, annotation: str, extractor: str) -> PluginType:
    return PluginType(
        key=key,
        annotations=(f"rextio_torch.types.{annotation}",),
        rust_type="RxtTorchTensor",
        conversion=BoundaryConversion(
            param_rust="pyo3::Bound<'py, pyo3::types::PyAny>",
            param_expr=f"{extractor}(py, &{{param}})?",
            return_rust="pyo3::Bound<'py, pyo3::types::PyAny>",
            return_expr="__rxttorch_materialize_tensor(py, {value})?",
        ),
        helpers=(_BOUNDARY_SUPPORT,),
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


__all__ = ["PLUGIN_TYPES", "plugin_type", "plugin_type_keys", "plugin_types"]
