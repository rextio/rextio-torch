"""Rust helpers for the materialized RxtTorchTensor boundary.

Uses tch 0.24.0 ``python-extension`` (``pyobject_unpack`` / ``pyobject_wrap``)
with PyO3 0.29-compatible raw pointer bridging. Extraction validates CPU,
float32, and rank; clone of the native wrapper uses ``shallow_clone``.
"""

from __future__ import annotations

# One exact-text module support block owned by both Phase A PluginTypes.
# Core deduplicates by exact text when both ranks appear in a signature.
_BOUNDARY_HELPERS = r"""struct RxtTorchTensor(tch::Tensor);

impl Clone for RxtTorchTensor {
    fn clone(&self) -> Self {
        Self(self.0.shallow_clone())
    }
}

fn __rxttorch_map_err(err: tch::TchError) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

fn __rxttorch_extract_f32_cpu(
    _py: pyo3::Python<'_>,
    value: &pyo3::Bound<'_, pyo3::types::PyAny>,
    expected_rank: i64,
) -> pyo3::PyResult<RxtTorchTensor> {
    let ptr = value.as_ptr();
    let tensor = unsafe { tch::Tensor::pyobject_unpack(ptr as *mut _) }
        .map_err(__rxttorch_map_err)?
        .ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err("rextio-torch: expected a torch.Tensor")
        })?;
    if tensor.device() != tch::Device::Cpu {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rextio-torch: expected a CPU tensor",
        ));
    }
    if tensor.kind() != tch::Kind::Float {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rextio-torch: expected a float32 tensor",
        ));
    }
    let rank = tensor.dim() as i64;
    if rank != expected_rank {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "rextio-torch: expected rank-{} tensor, got rank {}",
            expected_rank, rank
        )));
    }
    Ok(RxtTorchTensor(tensor))
}

fn __rxttorch_extract_f32_cpu_2d(
    py: pyo3::Python<'_>,
    value: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> pyo3::PyResult<RxtTorchTensor> {
    __rxttorch_extract_f32_cpu(py, value, 2)
}

fn __rxttorch_extract_f32_cpu_1d(
    py: pyo3::Python<'_>,
    value: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> pyo3::PyResult<RxtTorchTensor> {
    __rxttorch_extract_f32_cpu(py, value, 1)
}

fn __rxttorch_extract_i64_cpu_1d(
    _py: pyo3::Python<'_>,
    value: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> pyo3::PyResult<RxtTorchTensor> {
    let ptr = value.as_ptr();
    let tensor = unsafe { tch::Tensor::pyobject_unpack(ptr as *mut _) }
        .map_err(__rxttorch_map_err)?
        .ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err("rextio-torch: expected a torch.Tensor")
        })?;
    if tensor.device() != tch::Device::Cpu {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rextio-torch: expected a CPU tensor",
        ));
    }
    if tensor.kind() != tch::Kind::Int64 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rextio-torch: expected an int64 tensor",
        ));
    }
    let rank = tensor.dim() as i64;
    if rank != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "rextio-torch: expected rank-1 tensor, got rank {}", rank
        )));
    }
    Ok(RxtTorchTensor(tensor))
}

fn __rxttorch_materialize_tensor(
    py: pyo3::Python<'_>,
    value: RxtTorchTensor,
) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::types::PyAny>> {
    let ptr = value.0.pyobject_wrap().map_err(__rxttorch_map_err)?;
    Ok(unsafe { pyo3::Bound::from_owned_ptr(py, ptr as *mut _) })
}"""


def boundary_helpers() -> str:
    """Return the type-owned boundary struct and extract/materialize helpers."""
    return _BOUNDARY_HELPERS


__all__ = ["boundary_helpers"]
