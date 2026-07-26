"""Rust helpers for Alpha AOT tensor operations.

Every operation uses fallible tch APIs and maps ``TchError`` to a Python
exception without panicking. The default helper variants retain their own
``tch::no_grad_guard()`` for legacy hosts and functions containing Python
callbacks. Core plugin API 1.7 may instead install one reviewed RAII guard for
the whole native function; those functions use distinct ``*_function_scoped``
helpers that deliberately omit the redundant per-operation guard.
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
FUNCTION_SCOPED_SUFFIX = "_function_scoped"


def function_scoped_call_name(
    call_name: str,
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return a distinct helper symbol when Core installed the function guard."""
    if not isinstance(function_scope_guard_active, bool):
        raise TypeError("function_scope_guard_active must be a bool")
    if function_scope_guard_active:
        return f"{call_name}{FUNCTION_SCOPED_SUFFIX}"
    return call_name


def _guard_statement(*, function_scope_guard_active: bool) -> str:
    """Return the legacy per-operation guard or no statement for scoped helpers."""
    if function_scope_guard_active:
        return ""
    return "    let _guard = tch::no_grad_guard();\n"


def linear_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fallible functional linear helper for the selected guard mode."""
    call_name = function_scoped_call_name(
        LINEAR,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
    weight: &RxtTorchTensor,
    bias: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = input
        .0
        .f_linear(&weight.0, Some(&bias.0))
        .map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def linear_no_bias_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the functional linear helper with ``bias=None`` for one guard mode."""
    call_name = function_scoped_call_name(
        LINEAR_NO_BIAS,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
    weight: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = input
        .0
        .f_linear(&weight.0, Option::<&tch::Tensor>::None)
        .map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def relu_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fallible ReLU helper for the selected guard mode."""
    call_name = function_scoped_call_name(
        RELU,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(input: &RxtTorchTensor) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = input.0.f_relu().map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def sigmoid_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fallible sigmoid helper for the selected guard mode."""
    call_name = function_scoped_call_name(
        SIGMOID,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(input: &RxtTorchTensor) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = input.0.f_sigmoid().map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def tanh_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fallible tanh helper for the selected guard mode."""
    call_name = function_scoped_call_name(
        TANH,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(input: &RxtTorchTensor) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = input.0.f_tanh().map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


_UNARY_OPERATIONS: dict[str, tuple[str, str]] = {
    "abs": (ABS, "f_abs"),
    "neg": (NEG, "f_neg"),
    "negative": (NEGATIVE, "f_negative"),
    "square": (SQUARE, "f_square"),
    "exp": (EXP, "f_exp"),
    "log": (LOG, "f_log"),
    "sqrt": (SQRT, "f_sqrt"),
}


def unary_call_name(
    operation: str,
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return the fixed helper symbol for one bounded unary operation."""
    try:
        call_name = _UNARY_OPERATIONS[operation][0]
    except KeyError as exc:
        raise ValueError(f"unsupported rextio-torch unary operation: {operation!r}") from exc
    return function_scoped_call_name(
        call_name,
        function_scope_guard_active=function_scope_guard_active,
    )


def unary_helper(
    operation: str,
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return a fallible unary helper for one bounded operation and guard mode."""
    try:
        _, fallible_method = _UNARY_OPERATIONS[operation]
    except KeyError as exc:
        raise ValueError(f"unsupported rextio-torch unary operation: {operation!r}") from exc
    call_name = unary_call_name(
        operation,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = input.0.{fallible_method}().map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def gelu_none_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fixed GELU ``approximate='none'`` helper for one guard mode."""
    call_name = function_scoped_call_name(
        GELU_NONE,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = input.0.f_gelu("none").map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def mean_dim1_keepfalse_helper(
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return mean(dim=1, keepdim=False) for the selected guard mode."""
    return reduction_helper(
        "mean",
        1,
        False,
        function_scope_guard_active=function_scope_guard_active,
    )


def sum_dim1_keepfalse_helper(
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return sum(dim=1, keepdim=False) for the selected guard mode."""
    return reduction_helper(
        "sum",
        1,
        False,
        function_scope_guard_active=function_scope_guard_active,
    )


def reduction_call_name(
    method: str,
    dim: int,
    keepdim: bool,
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return the fixed helper symbol for one proved static reduction."""
    if method not in {"mean", "sum"} or dim not in {0, 1} or not isinstance(keepdim, bool):
        raise ValueError("rextio-torch reduction helper requires mean/sum, dim 0/1, bool keepdim")
    keep_token = "true" if keepdim else "false"
    return function_scoped_call_name(
        f"__rxttorch_{method}_dim{dim}_keepdim_{keep_token}",
        function_scope_guard_active=function_scope_guard_active,
    )


def reduction_helper(
    method: str,
    dim: int,
    keepdim: bool,
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return a fallible helper for one static reduction and guard mode."""
    call_name = reduction_call_name(
        method,
        dim,
        keepdim,
        function_scope_guard_active=function_scope_guard_active,
    )
    keep_token = "true" if keepdim else "false"
    if method == "mean":
        operation = f".f_mean_dim({dim}i64, {keep_token}, None)"
    else:
        operation = f".f_sum_dim_intlist({dim}i64, {keep_token}, None)"
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = input
        .0
        {operation}
        .map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def add_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fallible elementwise add helper for the selected guard mode."""
    call_name = function_scoped_call_name(
        ADD,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = left.0.f_add(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def mul_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fallible elementwise multiply helper for one guard mode."""
    call_name = function_scoped_call_name(
        MUL,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = left.0.f_mul(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def sub_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fallible elementwise subtraction helper for one guard mode."""
    call_name = function_scoped_call_name(
        SUB,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = left.0.f_sub(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def div_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fallible elementwise true-division helper for one guard mode."""
    call_name = function_scoped_call_name(
        DIV,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = left.0.f_div(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def matmul_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return the fallible rank-1/2 matmul helper for the selected guard mode."""
    call_name = function_scoped_call_name(
        MATMUL,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    left: &RxtTorchTensor,
    right: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = left.0.f_matmul(&right.0).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def softmax_dim1_helper(*, function_scope_guard_active: bool = False) -> str:
    """Return softmax(dim=1) for the selected guard mode."""
    return softmax_helper(
        1,
        function_scope_guard_active=function_scope_guard_active,
    )


def softmax_call_name(
    dim: int,
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return the fixed helper symbol for one proved softmax dimension."""
    if dim not in {0, 1}:
        raise ValueError("rextio-torch softmax helper requires dim 0 or 1")
    return function_scoped_call_name(
        f"__rxttorch_softmax_dim{dim}",
        function_scope_guard_active=function_scope_guard_active,
    )


def softmax_helper(
    dim: int,
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return a fallible softmax helper for one dimension and guard mode."""
    call_name = softmax_call_name(
        dim,
        function_scope_guard_active=function_scope_guard_active,
    )
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
    let out = input.0.f_softmax({dim}i64, None).map_err(__rxttorch_map_err)?;
    Ok(RxtTorchTensor(out))
}}"""


def argmax_dim1_keepfalse_helper(
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return argmax(dim=1, keepdim=False) for the selected guard mode."""
    return argmax_helper(
        1,
        False,
        expected_rank=1,
        function_scope_guard_active=function_scope_guard_active,
    )


def argmax_call_name(
    dim: int,
    keepdim: bool,
    *,
    function_scope_guard_active: bool = False,
) -> str:
    """Return the fixed helper symbol for one proved argmax variant."""
    if dim not in {0, 1} or not isinstance(keepdim, bool):
        raise ValueError("rextio-torch argmax helper requires dim 0/1 and bool keepdim")
    keep_token = "true" if keepdim else "false"
    return function_scoped_call_name(
        f"__rxttorch_argmax_dim{dim}_keepdim_{keep_token}",
        function_scope_guard_active=function_scope_guard_active,
    )


def argmax_helper(
    dim: int,
    keepdim: bool,
    *,
    expected_rank: int,
    function_scope_guard_active: bool = False,
) -> str:
    """Return a fallible argmax helper with an exact output boundary."""
    if expected_rank != 1:
        raise ValueError("rextio-torch currently exposes only int64 CPU rank-1 argmax")
    call_name = argmax_call_name(
        dim,
        keepdim,
        function_scope_guard_active=function_scope_guard_active,
    )
    keep_token = "true" if keepdim else "false"
    guard = _guard_statement(
        function_scope_guard_active=function_scope_guard_active
    )
    return f"""fn {call_name}(
    input: &RxtTorchTensor,
) -> pyo3::PyResult<RxtTorchTensor> {{
{guard}\
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
    "FUNCTION_SCOPED_SUFFIX",
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
    "function_scoped_call_name",
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
