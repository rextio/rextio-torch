"""Rust helpers for Phase A tensor operations (linear, ReLU, mean).

Every operation uses fallible tch APIs under ``no_grad`` and maps ``TchError``
to a Python exception without panicking.
"""

from __future__ import annotations

LINEAR = "__rxttorch_linear"
RELU = "__rxttorch_relu"
MEAN_DIM1_KEEPFALSE = "__rxttorch_mean_dim1_keepdim_false"


def linear_helper() -> str:
    """Return the no-grad fallible functional linear helper."""
    return r"""fn __rxttorch_linear(
    input: &RxtTorchTensor,
    weight: &RxtTorchTensor,
    bias: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input
        .0
        .f_linear(&weight.0, Some(&bias.0))
        .map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def relu_helper() -> str:
    """Return the no-grad fallible ReLU helper."""
    return r"""fn __rxttorch_relu(input: &RxtTorchTensor) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input.0.f_relu().map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def mean_dim1_keepfalse_helper() -> str:
    """Return the no-grad fallible mean(dim=1, keepdim=False) helper."""
    return r"""fn __rxttorch_mean_dim1_keepdim_false(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input
        .0
        .f_mean_dim(1i64, false, None)
        .map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


__all__ = [
    "LINEAR",
    "MEAN_DIM1_KEEPFALSE",
    "RELU",
    "linear_helper",
    "mean_dim1_keepfalse_helper",
    "relu_helper",
]
