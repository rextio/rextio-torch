"""Rust helpers for Alpha AOT tensor operations.

Every operation uses fallible tch APIs under ``no_grad`` and maps ``TchError``
to a Python exception without panicking.
"""

from __future__ import annotations

LINEAR = "__rxttorch_linear"
LINEAR_NO_BIAS = "__rxttorch_linear_no_bias"
RELU = "__rxttorch_relu"
SIGMOID = "__rxttorch_sigmoid"
TANH = "__rxttorch_tanh"
ABS = "__rxttorch_abs"
NEG = "__rxttorch_neg"
NEGATIVE = "__rxttorch_negative"
SQUARE = "__rxttorch_square"
EXP = "__rxttorch_exp"
LOG = "__rxttorch_log"
SQRT = "__rxttorch_sqrt"
GELU_NONE = "__rxttorch_gelu_none"
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


def linear_no_bias_helper() -> str:
    """Return the no-grad fallible functional linear helper with ``bias=None``."""
    return r"""fn __rxttorch_linear_no_bias(
    input: &RxtTorchTensor,
    weight: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input
        .0
        .f_linear(&weight.0, Option::<&tch::Tensor>::None)
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


_UNARY_OPERATIONS: dict[str, tuple[str, str]] = {
    "abs": (ABS, "f_abs"),
    "neg": (NEG, "f_neg"),
    "negative": (NEGATIVE, "f_negative"),
    "square": (SQUARE, "f_square"),
    "exp": (EXP, "f_exp"),
    "log": (LOG, "f_log"),
    "sqrt": (SQRT, "f_sqrt"),
}


def unary_call_name(operation: str) -> str:
    """Return the fixed helper symbol for one bounded unary operation."""
    try:
        return _UNARY_OPERATIONS[operation][0]
    except KeyError as exc:
        raise ValueError(f"unsupported rextio-torch unary operation: {operation!r}") from exc


def unary_helper(operation: str) -> str:
    """Return a fallible no-grad helper for one bounded unary operation."""
    try:
        call_name, fallible_method = _UNARY_OPERATIONS[operation]
    except KeyError as exc:
        raise ValueError(f"unsupported rextio-torch unary operation: {operation!r}") from exc
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
    let _guard = tch::no_grad_guard();
    let out = input.0.{fallible_method}().map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def gelu_none_helper() -> str:
    """Return the fixed fallible no-grad GELU ``approximate='none'`` helper."""
    return r"""fn __rxttorch_gelu_none(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {
    let _guard = tch::no_grad_guard();
    let out = input.0.f_gelu("none").map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}"""


def mean_dim1_keepfalse_helper() -> str:
    """Return the no-grad fallible mean(dim=1, keepdim=False) helper."""
    return reduction_helper("mean", 1, False)


def sum_dim1_keepfalse_helper() -> str:
    """Return the no-grad fallible sum(dim=1, keepdim=False) helper."""
    return reduction_helper("sum", 1, False)


def reduction_call_name(method: str, dim: int, keepdim: bool) -> str:
    """Return the fixed helper symbol for one proved static reduction."""
    if method not in {"mean", "sum"} or dim not in {0, 1} or not isinstance(keepdim, bool):
        raise ValueError("rextio-torch reduction helper requires mean/sum, dim 0/1, bool keepdim")
    keep_token = "true" if keepdim else "false"
    return f"__rxttorch_{method}_dim{dim}_keepdim_{keep_token}"


def reduction_helper(method: str, dim: int, keepdim: bool) -> str:
    """Return a fallible no-grad helper for one static mean/sum variant."""
    call_name = reduction_call_name(method, dim, keepdim)
    keep_token = "true" if keepdim else "false"
    if method == "mean":
        operation = f".f_mean_dim({dim}i64, {keep_token}, None)"
    else:
        operation = f".f_sum_dim_intlist({dim}i64, {keep_token}, None)"
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
    let _guard = tch::no_grad_guard();
    let out = input
        .0
        {operation}
        .map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


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
    return softmax_helper(1)


def softmax_call_name(dim: int) -> str:
    """Return the fixed helper symbol for one proved softmax dimension."""
    if dim not in {0, 1}:
        raise ValueError("rextio-torch softmax helper requires dim 0 or 1")
    return f"__rxttorch_softmax_dim{dim}"


def softmax_helper(dim: int) -> str:
    """Return a fallible no-grad softmax helper for one static dimension."""
    call_name = softmax_call_name(dim)
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
    let _guard = tch::no_grad_guard();
    let out = input.0.f_softmax({dim}i64, None).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def argmax_dim1_keepfalse_helper() -> str:
    """Return the no-grad fallible argmax(dim=1, keepdim=False) helper."""
    return argmax_helper(1, False, expected_rank=1)


def argmax_call_name(dim: int, keepdim: bool) -> str:
    """Return the fixed helper symbol for one proved argmax variant."""
    if dim not in {0, 1} or not isinstance(keepdim, bool):
        raise ValueError("rextio-torch argmax helper requires dim 0/1 and bool keepdim")
    keep_token = "true" if keepdim else "false"
    return f"__rxttorch_argmax_dim{dim}_keepdim_{keep_token}"


def argmax_helper(dim: int, keepdim: bool, *, expected_rank: int) -> str:
    """Return a fallible no-grad argmax helper with an exact output boundary."""
    if expected_rank != 1:
        raise ValueError("rextio-torch currently exposes only int64 CPU rank-1 argmax")
    call_name = argmax_call_name(dim, keepdim)
    keep_token = "true" if keepdim else "false"
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
    let _guard = tch::no_grad_guard();
    let out = input.0.f_argmax({dim}i64, {keep_token}).map_err(__rxttorch_map_err)?;
    if out.device() != tch::Device::Cpu
        || out.kind() != tch::Kind::Int64
        || out.dim() != {expected_rank}
    {{
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "rextio-torch: argmax classification result violated CPU int64 rank-1 boundary",
        ));
    }}
    Ok(RxtTorchTensor(out))
}}"""


__all__ = [
    "ABS",
    "ADD",
    "ARGMAX_DIM1_KEEPFALSE",
    "DIV",
    "EXP",
    "GELU_NONE",
    "LINEAR",
    "LINEAR_NO_BIAS",
    "LOG",
    "MATMUL",
    "MEAN_DIM1_KEEPFALSE",
    "MUL",
    "NEG",
    "NEGATIVE",
    "RELU",
    "SIGMOID",
    "SOFTMAX_DIM1",
    "SQRT",
    "SQUARE",
    "SUM_DIM1_KEEPFALSE",
    "SUB",
    "TANH",
    "add_helper",
    "argmax_call_name",
    "argmax_helper",
    "argmax_dim1_keepfalse_helper",
    "div_helper",
    "gelu_none_helper",
    "linear_helper",
    "linear_no_bias_helper",
    "matmul_helper",
    "mean_dim1_keepfalse_helper",
    "mul_helper",
    "relu_helper",
    "reduction_call_name",
    "reduction_helper",
    "sigmoid_helper",
    "softmax_call_name",
    "softmax_helper",
    "softmax_dim1_helper",
    "sum_dim1_keepfalse_helper",
    "sub_helper",
    "tanh_helper",
    "unary_call_name",
    "unary_helper",
]
