"""Annotation vocabulary stays import-free and maps to plugin type keys."""

from __future__ import annotations

import rextio_torch.types as types
from rextio_torch.plugin_types import PLUGIN_TYPES, plugin_type, plugin_types


def test_marker_classes_exist() -> None:
    assert types.TensorF32Cpu2D is not None
    assert types.TensorF32Cpu1D is not None
    assert types.TensorF32Cpu2D is not types.TensorF32Cpu1D


def test_plugin_types_registry_order_and_keys() -> None:
    registry = plugin_types()
    assert registry is PLUGIN_TYPES
    assert [item.key for item in registry] == [
        "rextio-torch/tensor-f32-cpu-2d",
        "rextio-torch/tensor-f32-cpu-1d",
        "rextio-torch/tensor-i64-cpu-1d",
    ]
    assert plugin_type("rextio-torch/tensor-f32-cpu-2d").annotations == (
        "rextio_torch.types.TensorF32Cpu2D",
    )
    assert plugin_type("rextio-torch/tensor-f32-cpu-1d").annotations == (
        "rextio_torch.types.TensorF32Cpu1D",
    )
    assert plugin_type("rextio-torch/tensor-i64-cpu-1d").annotations == (
        "rextio_torch.types.TensorI64Cpu1D",
    )


def test_extractors_are_rank_specific() -> None:
    by_key = {item.key: item for item in PLUGIN_TYPES}
    assert "__rxttorch_extract_f32_cpu_2d" in by_key[
        "rextio-torch/tensor-f32-cpu-2d"
    ].conversion.param_expr
    assert "__rxttorch_extract_f32_cpu_1d" in by_key[
        "rextio-torch/tensor-f32-cpu-1d"
    ].conversion.param_expr
    assert "__rxttorch_extract_i64_cpu_1d" in by_key[
        "rextio-torch/tensor-i64-cpu-1d"
    ].conversion.param_expr
