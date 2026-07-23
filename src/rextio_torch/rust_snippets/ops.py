"""Rust helpers for Alpha AOT tensor operations.

Every operation uses fallible tch APIs under ``no_grad`` and maps ``TchError``
to a Python exception without panicking.
"""

from __future__ import annotations

LINEAR = "__rxttorch_linear"
RELU = "__rxttorch_relu"
SIGMOID = "__rxttorch_sigmoid"
TANH = "__rxttorch_tanh"
MEAN_DIM1_KEEPFALSE = "__rxttorch_mean_dim1_keepdim_false"
SUM_DIM1_KEEPFALSE = "__rxttorch_sum_dim1_keepdim_false"
ADD = "__rxttorch_add"
MUL = "__rxttorch_mul"
SUB = "__rxttorch_sub"
DIV = "__rxttorch_div"
MATMUL = "__rxttorch_matmul"
SOFTMAX_DIM1 = "__rxttorch_softmax_dim1"
ARGMAX_DIM1_KEEPFALSE = "__rxttorch_argmax_dim1_keepdim_false"


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


def sigmoid_helper() -> str:
    """Return the no-grad fallible sigmoid helper."""
    return r"""fn __rxttorch_sigmoid(input: &RxtTorchTensor) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input.0.f_sigmoid().map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def tanh_helper() -> str:
    """Return the no-grad fallible tanh helper."""
    return r"""fn __rxttorch_tanh(input: &RxtTorchTensor) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input.0.f_tanh().map_err(__rxttorch_map_err)?;
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


def sum_dim1_keepfalse_helper() -> str:
    """Return the no-grad fallible sum(dim=1, keepdim=False) helper."""
    return r"""fn __rxttorch_sum_dim1_keepdim_false(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input
        .0
        .f_sum_dim_intlist(1i64, false, None)
        .map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def add_helper() -> str:
    """Return the no-grad fallible elementwise add helper (includes broadcast)."""
    return r"""fn __rxttorch_add(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = left.0.f_add(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def mul_helper() -> str:
    """Return the no-grad fallible elementwise multiply helper (includes broadcast)."""
    return r"""fn __rxttorch_mul(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = left.0.f_mul(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def sub_helper() -> str:
    """Return the no-grad fallible elementwise subtraction helper."""
    return r"""fn __rxttorch_sub(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = left.0.f_sub(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def div_helper() -> str:
    """Return the no-grad fallible elementwise true-division helper."""
    return r"""fn __rxttorch_div(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = left.0.f_div(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def matmul_helper() -> str:
    """Return the no-grad fallible rank-2 matmul helper."""
    return r"""fn __rxttorch_matmul(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = left.0.f_matmul(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def softmax_dim1_helper() -> str:
    """Return the no-grad fallible softmax(dim=1) helper."""
    return r"""fn __rxttorch_softmax_dim1(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input.0.f_softmax(1i64, None).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def argmax_dim1_keepfalse_helper() -> str:
    """Return the no-grad fallible argmax(dim=1, keepdim=False) helper."""
    return r"""fn __rxttorch_argmax_dim1_keepdim_false(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input.0.f_argmax(1i64, false).map_err(__rxttorch_map_err)?;
    if out.device() != tch::Device::Cpu || out.kind() != tch::Kind::Int64 || out.dim() != 1 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "rextio-torch: argmax classification result violated CPU int64 rank-1 boundary",
        ));
    }
    Ok(RxtTorchTensor(out))
}"""


__all__ = [
    "ADD",
    "ARGMAX_DIM1_KEEPFALSE",
    "DIV",
    "LINEAR",
    "MATMUL",
    "MEAN_DIM1_KEEPFALSE",
    "MUL",
    "RELU",
    "SIGMOID",
    "SOFTMAX_DIM1",
    "SUM_DIM1_KEEPFALSE",
    "SUB",
    "TANH",
    "add_helper",
    "argmax_dim1_keepfalse_helper",
    "div_helper",
    "linear_helper",
    "matmul_helper",
    "mean_dim1_keepfalse_helper",
    "mul_helper",
    "relu_helper",
    "sigmoid_helper",
    "softmax_dim1_helper",
    "sum_dim1_keepfalse_helper",
    "sub_helper",
    "tanh_helper",
]
