"""Real-Cargo vertical slice: Python for/if around Alpha AOT tch ops.

Certifies that scalar runtime control flow lowers to Rust control flow around
native tch helpers. A second function in the same serialized Cargo project
exercises functional/method matmul, same-rank add/multiply, sum, tanh, and
rank-1 activation chaining, without paying for another build. Evidence includes
native routes, numeric equivalence, dtype/device/rank, no-grad, non-mutation,
and fail-closed boundary rejects.

Build env requires CPython 3.11 + torch 2.11.0 via ``LIBTORCH_USE_PYTORCH=1``
(never ``LIBTORCH_BYPASS_VERSION_CHECK``).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from rextio.plugins.testing import CertifiedProject, build_certification_project

from rextio_torch.diagnostics import RUNTIME_ERRORS

pytestmark = [
    pytest.mark.needs_cargo,
    pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="real-Cargo certification requires cargo on PATH",
    ),
]

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PLUGIN_ROOT / ".venv" / "bin" / "python"

# Named temps: core cannot claim method calls whose receiver is a bare BinOp.
KERNELS = """
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D, TensorI64Cpu1D
import torch


def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
    depth: int,
    phase: int,
) -> TensorF32Cpu1D:
    hidden = x
    for layer in range(depth):
        if (layer + phase) % 2 == 0:
            # Distinct temps per branch: core scopes let-bindings per arm.
            even = hidden @ weight + bias
            hidden = even.relu()
        else:
            odd = hidden @ weight + bias
            hidden = odd.sigmoid()
    return hidden.mean(dim=1, keepdim=False)


def expanded_surface(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    offset: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    hidden = torch.matmul(x, weight)
    hidden = hidden.tanh()
    hidden = hidden + hidden
    hidden = hidden.matmul(weight)
    reduced = hidden.sum(dim=1, keepdim=False)
    reduced = reduced + offset
    return reduced.relu().sigmoid().tanh()


def multiply_surface(
    left: TensorF32Cpu2D,
    right: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
    vector: TensorF32Cpu1D,
) -> TensorF32Cpu2D:
    same_rank_2d = left * right
    same_rank_1d = bias * vector
    broadcast_forward = same_rank_2d * same_rank_1d
    return same_rank_1d * broadcast_forward


def classify(logits: TensorF32Cpu2D) -> TensorI64Cpu1D:
    probabilities = logits.softmax(dim=1)
    return probabilities.argmax(dim=1, keepdim=False)


def cpu_surface_followup(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
    divisor: TensorF32Cpu1D,
) -> TensorF32Cpu2D:
    omitted = torch.nn.functional.linear(x, weight)
    positional_none = torch.nn.functional.linear(omitted, weight, None)
    hidden = torch.nn.functional.linear(positional_none, weight, bias=None)
    activated = torch.relu(hidden)
    gated = torch.sigmoid(activated)
    shifted = gated - bias
    scaled = shifted / divisor
    totals = torch.sum(scaled, 0, keepdim=True)
    averaged = torch.mean(totals, 1, keepdim=True)
    probabilities = torch.softmax(averaged, 1)
    return torch.tanh(probabilities)


def arithmetic_followup(
    left: TensorF32Cpu2D,
    right: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
    vector: TensorF32Cpu1D,
) -> TensorF32Cpu2D:
    same_sub_2d = left - right
    same_sub_1d = bias - vector
    broadcast_sub_forward = same_sub_2d - same_sub_1d
    broadcast_sub_reverse = same_sub_1d - same_sub_2d
    same_div_2d = left / right
    same_div_1d = bias / vector
    broadcast_div_forward = same_div_2d / same_div_1d
    broadcast_div_reverse = same_div_1d / same_div_2d
    combined_sub = broadcast_sub_forward - broadcast_sub_reverse
    return combined_sub / (broadcast_div_forward + broadcast_div_reverse)


def functional_arithmetic_aliases(
    left: TensorF32Cpu2D,
    right: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
    vector: TensorF32Cpu1D,
) -> TensorF32Cpu2D:
    added = torch.add(left, right)
    subtracted = torch.sub(added, bias)
    multiplied = torch.mul(vector, subtracted)
    return torch.div(multiplied, right)


def mixed_rank_matmul(
    matrix: TensorF32Cpu2D,
    left_vector: TensorF32Cpu1D,
    right_vector: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    binop_mv = matrix @ right_vector
    binop_vm = left_vector @ matrix
    function_mv = torch.matmul(matrix, right_vector)
    function_vm = torch.matmul(left_vector, matrix)
    method_mv = matrix.matmul(right_vector)
    method_vm = left_vector.matmul(matrix)
    return binop_mv + binop_vm + function_mv + function_vm + method_mv + method_vm


def unary_abs(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return torch.abs(x)


def unary_neg(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return x.neg()


def unary_negative(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return torch.negative(x)


def unary_square(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return x.square()


def unary_exp(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return torch.exp(x)


def unary_log(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return x.log()


def unary_sqrt(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return torch.sqrt(x)


def gelu_default(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return torch.nn.functional.gelu(x)


def gelu_explicit_none(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return torch.nn.functional.gelu(x, approximate="none")


def functional_classify(logits: TensorF32Cpu2D) -> TensorI64Cpu1D:
    probabilities = torch.softmax(logits, 0)
    return torch.argmax(probabilities, dim=1)


def vector_classify(logits: TensorF32Cpu1D) -> TensorI64Cpu1D:
    probabilities = torch.softmax(logits, dim=0)
    return torch.argmax(probabilities, 0, keepdim=True)
"""


def _configure_libtorch_env() -> None:
    """Point tch/PyO3 at this project's CPython 3.11 + torch 2.11 venv."""
    venv_dir = PLUGIN_ROOT / ".venv"
    venv_bin = venv_dir / "bin"
    if not VENV_PYTHON.is_file():
        pytest.skip(f"dedicated venv python missing: {VENV_PYTHON}")
    os.environ["VIRTUAL_ENV"] = str(venv_dir)
    os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["PYO3_PYTHON"] = str(VENV_PYTHON)
    os.environ["LIBTORCH_USE_PYTORCH"] = "1"
    os.environ.pop("LIBTORCH_BYPASS_VERSION_CHECK", None)
    assert "LIBTORCH_BYPASS_VERSION_CHECK" not in os.environ
    import subprocess

    probe = subprocess.run(
        ["python", "-c", "import torch; print(torch.__version__.split('+')[0])"],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert probe.returncode == 0, probe.stderr
    version = probe.stdout.strip()
    assert version == "2.11.0", (
        f"torch-sys build python must be torch 2.11.0; `python` on PATH reports {version!r}"
    )


def _import_torch():
    import torch

    return torch


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> CertifiedProject:
    _configure_libtorch_env()
    _import_torch()

    root = tmp_path_factory.mktemp("torch_alpha_aot")
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-torch"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "torch_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNELS, encoding="utf-8")
    return build_certification_project(root)


def _route_of(project: CertifiedProject, qualname: str) -> dict:
    report = json.loads(
        (project.project_root / ".rextio" / "reports" / "check.json").read_text(encoding="utf-8")
    )
    for module in report["modules"]:
        for function in module["functions"]:
            if function["qualname"] == qualname:
                return function
    raise AssertionError(f"{qualname} not found in check.json")


def _tensor_equal(left: object, right: object) -> bool:
    torch = _import_torch()
    if type(left) is not torch.Tensor or type(right) is not torch.Tensor:
        return False
    if left.dtype != right.dtype or left.device != right.device or left.shape != right.shape:
        return False
    if left.requires_grad != right.requires_grad:
        return False
    return bool(torch.allclose(left, right, rtol=1e-5, atol=1e-6, equal_nan=True))


def _args_unmutated(left: object, right: object) -> bool:
    """Compare post-call args: tensors by value, scalars by identity equality."""
    torch = _import_torch()
    if type(left) is torch.Tensor or type(right) is torch.Tensor:
        return _tensor_equal(left, right)
    return left == right


def _copy_tensor_args(args: tuple[object, ...]) -> tuple[object, ...]:
    torch = _import_torch()
    copies: list[object] = []
    for arg in args:
        if type(arg) is torch.Tensor:
            copies.append(arg.detach().clone())
        else:
            copies.append(arg)
    return tuple(copies)


def _eager_inference(x, weight, bias, depth: int, phase: int):
    hidden = x
    for layer in range(depth):
        if (layer + phase) % 2 == 0:
            even = hidden @ weight + bias
            hidden = even.relu()
        else:
            odd = hidden @ weight + bias
            hidden = odd.sigmoid()
    return hidden.mean(dim=1, keepdim=False)


def _eager_expanded_surface(x, weight, offset):
    hidden = _import_torch().matmul(x, weight)
    hidden = hidden.tanh()
    hidden = hidden + hidden
    hidden = hidden.matmul(weight)
    reduced = hidden.sum(dim=1, keepdim=False)
    reduced = reduced + offset
    return reduced.relu().sigmoid().tanh()


def _eager_classify(logits):
    return logits.softmax(dim=1).argmax(dim=1, keepdim=False)


def _eager_multiply_surface(left, right, bias, vector):
    same_rank_2d = left * right
    same_rank_1d = bias * vector
    broadcast_forward = same_rank_2d * same_rank_1d
    return same_rank_1d * broadcast_forward


def _eager_cpu_surface_followup(x, weight, bias, divisor):
    torch = _import_torch()
    omitted = torch.nn.functional.linear(x, weight)
    positional_none = torch.nn.functional.linear(omitted, weight, None)
    hidden = torch.nn.functional.linear(positional_none, weight, bias=None)
    activated = torch.relu(hidden)
    gated = torch.sigmoid(activated)
    shifted = gated - bias
    scaled = shifted / divisor
    totals = torch.sum(scaled, 0, keepdim=True)
    averaged = torch.mean(totals, 1, keepdim=True)
    probabilities = torch.softmax(averaged, 1)
    return torch.tanh(probabilities)


def _eager_arithmetic_followup(left, right, bias, vector):
    same_sub_2d = left - right
    same_sub_1d = bias - vector
    broadcast_sub_forward = same_sub_2d - same_sub_1d
    broadcast_sub_reverse = same_sub_1d - same_sub_2d
    same_div_2d = left / right
    same_div_1d = bias / vector
    broadcast_div_forward = same_div_2d / same_div_1d
    broadcast_div_reverse = same_div_1d / same_div_2d
    combined_sub = broadcast_sub_forward - broadcast_sub_reverse
    return combined_sub / (broadcast_div_forward + broadcast_div_reverse)


def _eager_functional_arithmetic_aliases(left, right, bias, vector):
    torch = _import_torch()
    added = torch.add(left, right)
    subtracted = torch.sub(added, bias)
    multiplied = torch.mul(vector, subtracted)
    return torch.div(multiplied, right)


def _eager_mixed_rank_matmul(matrix, left_vector, right_vector):
    torch = _import_torch()
    binop_mv = matrix @ right_vector
    binop_vm = left_vector @ matrix
    function_mv = torch.matmul(matrix, right_vector)
    function_vm = torch.matmul(left_vector, matrix)
    method_mv = matrix.matmul(right_vector)
    method_vm = left_vector.matmul(matrix)
    return binop_mv + binop_vm + function_mv + function_vm + method_mv + method_vm


def _eager_unary(name: str, value):
    torch = _import_torch()
    operations = {
        "unary_abs": torch.abs,
        "unary_neg": lambda tensor: tensor.neg(),
        "unary_negative": torch.negative,
        "unary_square": lambda tensor: tensor.square(),
        "unary_exp": torch.exp,
        "unary_log": lambda tensor: tensor.log(),
        "unary_sqrt": torch.sqrt,
    }
    return operations[name](value)


def _assert_special_value_equivalence(actual: object, expected: object) -> None:
    """Compare IEEE classes and signed zero without promising NaN payload bits."""
    torch = _import_torch()
    assert type(actual) is torch.Tensor
    assert type(expected) is torch.Tensor
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    zero_mask = expected == 0
    assert torch.equal(torch.signbit(actual[zero_mask]), torch.signbit(expected[zero_mask]))
    finite_mask = torch.isfinite(expected)
    assert torch.allclose(
        actual[finite_mask],
        expected[finite_mask],
        rtol=1e-5,
        atol=1e-6,
    )


def _eager_gelu(value, *, explicit_none: bool):
    functional = _import_torch().nn.functional
    if explicit_none:
        return functional.gelu(value, approximate="none")
    return functional.gelu(value)


def _eager_functional_classify(logits):
    torch = _import_torch()
    return torch.argmax(torch.softmax(logits, 0), dim=1)


def _eager_vector_classify(logits):
    torch = _import_torch()
    return torch.argmax(torch.softmax(logits, dim=0), 0, keepdim=True)


def test_alpha_aot_control_flow_real_cargo(project: CertifiedProject) -> None:
    """Serialized build: control-flow native route, equivalence, contracts, rejects."""
    torch = _import_torch()
    assert torch.__version__.startswith("2.11.")
    assert sys.version_info[:2] == (3, 11)

    build = json.loads(
        (project.project_root / ".rextio" / "reports" / "build.json").read_text(encoding="utf-8")
    )
    assert build["status"] == "built"
    native_build = build.get("native_build") or {}
    assert native_build.get("status") == "built"

    record = _route_of(project, "torch_app.kernels.inference")
    assert record["native_status"] == "accepted"
    assert record["route"] == "native-plugin:rextio-torch"
    claims = record.get("plugin_claims") or []
    claim_rules = {claim["rule_id"] for claim in claims}
    assert "rextio-torch/tensor-matmul-f32-cpu-2d" in claim_rules
    assert "rextio-torch/tensor-add-f32-cpu-2d-1d-broadcast" in claim_rules
    assert "rextio-torch/tensor-relu-f32-cpu-2d" in claim_rules
    assert "rextio-torch/tensor-sigmoid-f32-cpu-2d" in claim_rules
    assert "rextio-torch/tensor-mean-dim1-f32-cpu-2d" in claim_rules

    expanded_record = _route_of(project, "torch_app.kernels.expanded_surface")
    assert expanded_record["native_status"] == "accepted"
    assert expanded_record["route"] == "native-plugin:rextio-torch"
    expanded_claims = expanded_record.get("plugin_claims") or []
    expanded_rules = [claim["rule_id"] for claim in expanded_claims]
    assert "rextio-torch/tensor-matmul-call-f32-cpu-2d" in expanded_rules
    assert "rextio-torch/tensor-tanh-f32-cpu-2d" in expanded_rules
    assert "rextio-torch/tensor-add-f32-cpu-same-rank" in expanded_rules
    assert "rextio-torch/tensor-sum-dim1-f32-cpu-2d" in expanded_rules
    assert "rextio-torch/tensor-relu-f32-cpu-1d" in expanded_rules
    assert "rextio-torch/tensor-sigmoid-f32-cpu-1d" in expanded_rules
    assert "rextio-torch/tensor-tanh-f32-cpu-1d" in expanded_rules
    assert expanded_rules.count("rextio-torch/tensor-matmul-call-f32-cpu-2d") == 2
    assert expanded_rules.count("rextio-torch/tensor-add-f32-cpu-same-rank") == 2

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "struct RxtTorchTensor" in rust
    assert "__rxttorch_matmul" in rust
    assert "__rxttorch_add" in rust
    assert "__rxttorch_relu" in rust
    assert "__rxttorch_sigmoid" in rust
    assert "__rxttorch_tanh" in rust
    assert "__rxttorch_mean_dim1_keepdim_false" in rust
    assert "__rxttorch_sum_dim1_keepdim_false" in rust
    assert "no_grad_guard" in rust
    # Python for/if must appear as Rust control flow, not only as tch helper calls.
    assert "for " in rust or "loop " in rust
    assert "if " in rust
    assert "LIBTORCH_BYPASS_VERSION_CHECK" not in os.environ

    torch.manual_seed(7)
    batch, width = 4, 3
    x = torch.randn(batch, width, dtype=torch.float32)
    weight = torch.randn(width, width, dtype=torch.float32)
    bias = torch.randn(width, dtype=torch.float32)
    depth = 3
    phase = 1
    x_snap = x.detach().clone()
    w_snap = weight.detach().clone()
    b_snap = bias.detach().clone()

    checker = project.equivalence_checker(
        "torch_app.kernels.inference",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    native_out = checker(x, weight, bias, depth, phase)

    assert type(native_out) is torch.Tensor
    assert native_out.device.type == "cpu"
    assert native_out.dtype == torch.float32
    assert native_out.dim() == 1
    assert tuple(native_out.shape) == (batch,)
    assert native_out.requires_grad is False

    eager = _eager_inference(x_snap, w_snap, b_snap, depth, phase)
    assert _tensor_equal(native_out, eager)

    assert torch.equal(x, x_snap)
    assert torch.equal(weight, w_snap)
    assert torch.equal(bias, b_snap)

    # Same serialized build exercises every expanded Alpha helper/rank lane:
    # functional + method matmul, same-rank add (2D and 1D), sum, tanh, and
    # rank-1 activation chaining.
    offset = torch.randn(batch, dtype=torch.float32)
    offset_snap = offset.detach().clone()
    expanded_checker = project.equivalence_checker(
        "torch_app.kernels.expanded_surface",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    expanded_out = expanded_checker(x, weight, offset)
    expanded_eager = _eager_expanded_surface(x_snap, w_snap, offset_snap)
    assert _tensor_equal(expanded_out, expanded_eager)
    assert expanded_out.device.type == "cpu"
    assert expanded_out.dtype == torch.float32
    assert expanded_out.dim() == 1
    assert tuple(expanded_out.shape) == (batch,)
    assert expanded_out.requires_grad is False
    assert torch.equal(x, x_snap)
    assert torch.equal(weight, w_snap)
    assert torch.equal(offset, offset_snap)

    x_exp_g = x_snap.detach().clone().requires_grad_(True)
    w_exp_g = w_snap.detach().clone().requires_grad_(True)
    offset_g = offset_snap.detach().clone().requires_grad_(True)
    with _native_mode(project, "native"):
        from torch_app.kernels import expanded_surface as expanded_fn

        expanded_grad_out = expanded_fn(x_exp_g, w_exp_g, offset_g)
    assert expanded_grad_out.requires_grad is False
    assert torch.allclose(expanded_grad_out, expanded_eager, rtol=1e-5, atol=1e-6)
    assert torch.equal(x_exp_g.detach(), x_snap)
    assert torch.equal(w_exp_g.detach(), w_snap)
    assert torch.equal(offset_g.detach(), offset_snap)

    multiply_record = _route_of(project, "torch_app.kernels.multiply_surface")
    assert multiply_record["native_status"] == "accepted"
    assert multiply_record["route"] == "native-plugin:rextio-torch"
    multiply_rules = [claim["rule_id"] for claim in multiply_record.get("plugin_claims") or []]
    assert multiply_rules.count("rextio-torch/tensor-mul-f32-cpu-same-rank") == 2
    assert multiply_rules.count("rextio-torch/tensor-mul-f32-cpu-2d-1d-broadcast") == 2
    assert "__rxttorch_mul" in rust
    assert "f_mul" in rust

    mul_left = torch.tensor(
        [[-0.0, float("inf"), float("nan")], [2.0, -3.0, 4.0]], dtype=torch.float32
    )
    mul_right = torch.tensor(
        [[2.0, -1.0, 1.0], [0.5, -2.0, float("inf")]], dtype=torch.float32
    )
    mul_bias = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
    mul_vector = torch.tensor([1.0, 1.0, -2.0], dtype=torch.float32)
    mul_left_snap = mul_left.detach().clone()
    mul_right_snap = mul_right.detach().clone()
    mul_bias_snap = mul_bias.detach().clone()
    mul_vector_snap = mul_vector.detach().clone()
    multiply_checker = project.equivalence_checker(
        "torch_app.kernels.multiply_surface",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    multiply_out = multiply_checker(mul_left, mul_right, mul_bias, mul_vector)
    multiply_eager = _eager_multiply_surface(
        mul_left_snap, mul_right_snap, mul_bias_snap, mul_vector_snap
    )
    assert _tensor_equal(multiply_out, multiply_eager)
    assert multiply_out.device.type == "cpu"
    assert multiply_out.dtype == torch.float32
    assert multiply_out.dim() == 2
    assert multiply_out.requires_grad is False
    assert _tensor_equal(mul_left, mul_left_snap)
    assert _tensor_equal(mul_right, mul_right_snap)
    assert _tensor_equal(mul_bias, mul_bias_snap)
    assert _tensor_equal(mul_vector, mul_vector_snap)
    assert torch.equal(torch.signbit(multiply_out), torch.signbit(multiply_eager))

    mul_left_g = mul_left_snap.detach().clone().requires_grad_(True)
    mul_right_g = mul_right_snap.detach().clone().requires_grad_(True)
    mul_bias_g = mul_bias_snap.detach().clone().requires_grad_(True)
    mul_vector_g = mul_vector_snap.detach().clone().requires_grad_(True)
    with _native_mode(project, "native"):
        from torch_app.kernels import multiply_surface

        multiply_grad_out = multiply_surface(mul_left_g, mul_right_g, mul_bias_g, mul_vector_g)
    assert multiply_grad_out.requires_grad is False
    assert _tensor_equal(multiply_grad_out, multiply_eager)
    assert _tensor_equal(mul_left_g.detach(), mul_left_snap)
    assert _tensor_equal(mul_right_g.detach(), mul_right_snap)

    # Concrete broadcast sizes are intentionally a runtime libtorch contract.
    with _native_mode(project, "native"):
        from torch_app.kernels import multiply_surface as multiply_boundary

        with pytest.raises(RuntimeError):
            multiply_boundary(
                torch.ones((2, 2)),
                torch.ones((2, 2)),
                torch.ones(3),
                torch.ones(3),
            )

    # Grad-requesting inputs still yield no-grad native output.
    x_g = x_snap.detach().clone().requires_grad_(True)
    w_g = w_snap.detach().clone().requires_grad_(True)
    b_g = b_snap.detach().clone().requires_grad_(True)
    with _native_mode(project, "native"):
        from torch_app.kernels import inference as inference_fn

        grad_out = inference_fn(x_g, w_g, b_g, depth, phase)
    assert grad_out.requires_grad is False
    assert torch.allclose(grad_out, eager, rtol=1e-5, atol=1e-6)
    assert torch.equal(x_g.detach(), x_snap)
    assert torch.equal(w_g.detach(), w_snap)
    assert torch.equal(b_g.detach(), b_snap)

    # Boundary rejects.
    w_ok = w_snap.detach().clone()
    b_ok = b_snap.detach().clone()
    with _native_mode(project, "native"):
        from torch_app.kernels import inference as inference_boundary

        x_f64 = torch.randn(batch, width, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"^rextio-torch: expected a float32 tensor$") as dtype_info:
            inference_boundary(x_f64, w_ok, b_ok, depth, phase)
        assert str(dtype_info.value) == RUNTIME_ERRORS["dtype"]

        x_rank1 = torch.randn(width, dtype=torch.float32)
        with pytest.raises(
            ValueError,
            match=r"^rextio-torch: expected rank-2 tensor, got rank 1$",
        ):
            inference_boundary(x_rank1, w_ok, b_ok, depth, phase)

    assert torch.equal(w_ok, w_snap)
    assert torch.equal(b_ok, b_snap)


def test_classification_head_real_cargo(project: CertifiedProject) -> None:
    """The method-only softmax→argmax head materializes int64 CPU rank-1 output."""
    torch = _import_torch()
    record = _route_of(project, "torch_app.kernels.classify")
    assert record["native_status"] == "accepted"
    assert record["route"] == "native-plugin:rextio-torch"
    claims = record.get("plugin_claims") or []
    assert [claim["rule_id"] for claim in claims] == [
        "rextio-torch/tensor-softmax-dim1-f32-cpu-2d",
        "rextio-torch/tensor-argmax-dim1-keepfalse-i64-cpu-1d",
    ]
    assert claims[-1]["result_type"] == "rextio-torch/tensor-i64-cpu-1d"

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "__rxttorch_softmax_dim1" in rust
    assert "__rxttorch_argmax_dim1_keepdim_false" in rust
    assert "f_softmax(1i64, None)" in rust
    assert "f_argmax(1i64, false)" in rust

    logits = torch.tensor(
        [[-3.0, 2.0, 2.0], [0.5, 0.4, 0.6], [100.0, -100.0, 1.0]], dtype=torch.float32
    )
    logits_snap = logits.detach().clone()
    checker = project.equivalence_checker(
        "torch_app.kernels.classify",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    native_out = checker(logits)
    eager = _eager_classify(logits_snap)
    assert _tensor_equal(native_out, eager)
    assert type(native_out) is torch.Tensor
    assert native_out.device.type == "cpu"
    assert native_out.dtype == torch.int64
    assert native_out.dim() == 1
    assert tuple(native_out.shape) == (3,)
    assert native_out.requires_grad is False
    assert torch.equal(logits, logits_snap)

    logits_grad = logits_snap.detach().clone().requires_grad_(True)
    with _native_mode(project, "native"):
        from torch_app.kernels import classify

        native_grad_out = classify(logits_grad)
    assert native_grad_out.requires_grad is False
    assert native_grad_out.dtype == torch.int64
    assert torch.equal(native_grad_out, eager)
    assert torch.equal(logits_grad.detach(), logits_snap)


def test_cpu_surface_followup_real_cargo(project: CertifiedProject) -> None:
    """Certify the bounded function/static-dim/sub/div/no-bias vertical slice."""
    torch = _import_torch()
    followup_record = _route_of(project, "torch_app.kernels.cpu_surface_followup")
    assert followup_record["native_status"] == "accepted"
    assert followup_record["route"] == "native-plugin:rextio-torch"
    followup_rules = [
        claim["rule_id"] for claim in followup_record.get("plugin_claims") or []
    ]
    assert followup_rules.count(
        "rextio-torch/functional-linear-none-f32-cpu-2d"
    ) == 3
    for rule_id in (
        "rextio-torch/function-relu-f32-cpu-rank1-2",
        "rextio-torch/function-sigmoid-f32-cpu-rank1-2",
        "rextio-torch/function-tanh-f32-cpu-rank1-2",
        "rextio-torch/tensor-sub-f32-cpu-2d-1d-broadcast",
        "rextio-torch/tensor-div-f32-cpu-2d-1d-broadcast",
        "rextio-torch/sum-static-dim-f32-cpu-rank1-2",
        "rextio-torch/mean-static-dim-f32-cpu-rank1-2",
        "rextio-torch/softmax-static-dim-f32-cpu-rank1-2",
    ):
        assert rule_id in followup_rules

    arithmetic_record = _route_of(project, "torch_app.kernels.arithmetic_followup")
    assert arithmetic_record["native_status"] == "accepted"
    assert arithmetic_record["route"] == "native-plugin:rextio-torch"
    arithmetic_rules = [
        claim["rule_id"] for claim in arithmetic_record.get("plugin_claims") or []
    ]
    assert arithmetic_rules.count("rextio-torch/tensor-sub-f32-cpu-same-rank") == 3
    assert (
        arithmetic_rules.count(
            "rextio-torch/tensor-sub-f32-cpu-2d-1d-broadcast"
        )
        == 2
    )
    assert arithmetic_rules.count("rextio-torch/tensor-div-f32-cpu-same-rank") == 3
    assert (
        arithmetic_rules.count(
            "rextio-torch/tensor-div-f32-cpu-2d-1d-broadcast"
        )
        == 2
    )

    alias_record = _route_of(
        project,
        "torch_app.kernels.functional_arithmetic_aliases",
    )
    assert alias_record["native_status"] == "accepted"
    assert alias_record["route"] == "native-plugin:rextio-torch"
    alias_rules = [
        claim["rule_id"] for claim in alias_record.get("plugin_claims") or []
    ]
    assert alias_rules == [
        "rextio-torch/function-add-f32-cpu-rank1-2",
        "rextio-torch/function-sub-f32-cpu-rank1-2",
        "rextio-torch/function-mul-f32-cpu-rank1-2",
        "rextio-torch/function-div-f32-cpu-rank1-2",
    ]

    mixed_record = _route_of(project, "torch_app.kernels.mixed_rank_matmul")
    assert mixed_record["native_status"] == "accepted"
    assert mixed_record["route"] == "native-plugin:rextio-torch"
    mixed_rules = [
        claim["rule_id"] for claim in mixed_record.get("plugin_claims") or []
    ]
    assert mixed_rules.count(
        "rextio-torch/tensor-matmul-f32-cpu-mixed-rank"
    ) == 2
    assert mixed_rules.count(
        "rextio-torch/tensor-matmul-call-f32-cpu-mixed-rank"
    ) == 4

    unary_contracts = (
        ("unary_abs", "rextio-torch/unary-abs-f32-cpu-rank1-2"),
        ("unary_neg", "rextio-torch/unary-neg-f32-cpu-rank1-2"),
        ("unary_negative", "rextio-torch/unary-negative-f32-cpu-rank1-2"),
        ("unary_square", "rextio-torch/unary-square-f32-cpu-rank1-2"),
        ("unary_exp", "rextio-torch/unary-exp-f32-cpu-rank1-2"),
        ("unary_log", "rextio-torch/unary-log-f32-cpu-rank1-2"),
        ("unary_sqrt", "rextio-torch/unary-sqrt-f32-cpu-rank1-2"),
    )
    for qualname, rule_id in unary_contracts:
        unary_record = _route_of(project, f"torch_app.kernels.{qualname}")
        assert unary_record["native_status"] == "accepted"
        assert unary_record["route"] == "native-plugin:rextio-torch"
        unary_claims = unary_record.get("plugin_claims") or []
        assert [claim["rule_id"] for claim in unary_claims] == [rule_id]
        assert unary_claims[0]["result_type"] == "rextio-torch/tensor-f32-cpu-1d"

    gelu_contracts = (
        ("gelu_default", False),
        ("gelu_explicit_none", True),
    )
    for qualname, _explicit_none in gelu_contracts:
        gelu_record = _route_of(project, f"torch_app.kernels.{qualname}")
        assert gelu_record["native_status"] == "accepted"
        assert gelu_record["route"] == "native-plugin:rextio-torch"
        gelu_claims = gelu_record.get("plugin_claims") or []
        assert [claim["rule_id"] for claim in gelu_claims] == [
            "rextio-torch/functional-gelu-none-f32-cpu-rank1-2"
        ]
        assert gelu_claims[0]["result_type"] == "rextio-torch/tensor-f32-cpu-1d"

    functional_record = _route_of(
        project, "torch_app.kernels.functional_classify"
    )
    vector_record = _route_of(project, "torch_app.kernels.vector_classify")
    for record in (functional_record, vector_record):
        assert record["native_status"] == "accepted"
        assert record["route"] == "native-plugin:rextio-torch"
        rules = [claim["rule_id"] for claim in record.get("plugin_claims") or []]
        assert rules == [
            "rextio-torch/softmax-static-dim-f32-cpu-rank1-2",
            "rextio-torch/argmax-static-dim-i64-cpu-rank1",
        ]

    rust = (
        project.project_root
        / ".rextio"
        / "generated"
        / "rust"
        / "src"
        / "lib.rs"
    ).read_text(encoding="utf-8")
    for symbol in (
        "__rxttorch_linear_no_bias",
        "__rxttorch_sub",
        "__rxttorch_div",
        "__rxttorch_sum_dim0_keepdim_true",
        "__rxttorch_mean_dim1_keepdim_true",
        "__rxttorch_softmax_dim0",
        "__rxttorch_argmax_dim0_keepdim_true",
        "__rxttorch_abs",
        "__rxttorch_neg",
        "__rxttorch_negative",
        "__rxttorch_square",
        "__rxttorch_exp",
        "__rxttorch_log",
        "__rxttorch_sqrt",
        "__rxttorch_gelu_none",
    ):
        assert symbol in rust
    assert "f_linear(&weight.0, Option::<&tch::Tensor>::None)" in rust
    assert "f_sub" in rust
    assert "f_div" in rust
    for fallible_method in (
        "f_abs",
        "f_neg",
        "f_negative",
        "f_square",
        "f_exp",
        "f_log",
        "f_sqrt",
    ):
        assert f".{fallible_method}()" in rust
    assert '.f_gelu("none")' in rust
    assert '.f_gelu("tanh")' not in rust
    assert "no_grad_guard" in rust

    torch.manual_seed(23)
    width = 3
    x = torch.randn((4, width), dtype=torch.float32)
    weight = torch.randn((width, width), dtype=torch.float32)
    bias = torch.tensor([0.25, -0.75, 1.5], dtype=torch.float32)
    divisor = torch.tensor([0.5, -1.25, 2.0], dtype=torch.float32)
    x_snap = x.detach().clone()
    weight_snap = weight.detach().clone()
    bias_snap = bias.detach().clone()
    divisor_snap = divisor.detach().clone()
    followup_checker = project.equivalence_checker(
        "torch_app.kernels.cpu_surface_followup",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    followup_out = followup_checker(x, weight, bias, divisor)
    followup_eager = _eager_cpu_surface_followup(
        x_snap, weight_snap, bias_snap, divisor_snap
    )
    assert _tensor_equal(followup_out, followup_eager)
    assert followup_out.device.type == "cpu"
    assert followup_out.dtype == torch.float32
    assert tuple(followup_out.shape) == (1, 1)
    assert followup_out.requires_grad is False
    assert torch.equal(x, x_snap)
    assert torch.equal(weight, weight_snap)
    assert torch.equal(bias, bias_snap)
    assert torch.equal(divisor, divisor_snap)

    with _native_mode(project, "native"):
        from torch_app.kernels import cpu_surface_followup

        followup_grad = cpu_surface_followup(
            x_snap.detach().clone().requires_grad_(True),
            weight_snap.detach().clone().requires_grad_(True),
            bias_snap.detach().clone().requires_grad_(True),
            divisor_snap.detach().clone().requires_grad_(True),
        )
    assert followup_grad.requires_grad is False
    assert torch.allclose(
        followup_grad,
        followup_eager,
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
    )

    left = torch.tensor(
        [[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=torch.float32
    )
    right = torch.tensor(
        [[2.0, 4.0, 8.0], [6.0, 10.0, 14.0]], dtype=torch.float32
    )
    arithmetic_bias = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    vector = torch.tensor([0.5, 1.25, 1.75], dtype=torch.float32)
    arithmetic_args = (left, right, arithmetic_bias, vector)
    arithmetic_snap = _copy_tensor_args(arithmetic_args)
    arithmetic_checker = project.equivalence_checker(
        "torch_app.kernels.arithmetic_followup",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    arithmetic_out = arithmetic_checker(*arithmetic_args)
    arithmetic_eager = _eager_arithmetic_followup(*arithmetic_snap)
    assert _tensor_equal(arithmetic_out, arithmetic_eager)
    assert arithmetic_out.device.type == "cpu"
    assert arithmetic_out.dtype == torch.float32
    assert tuple(arithmetic_out.shape) == (2, 3)
    assert arithmetic_out.requires_grad is False
    for actual, snapshot in zip(arithmetic_args, arithmetic_snap, strict=True):
        assert _tensor_equal(actual, snapshot)

    alias_checker = project.equivalence_checker(
        "torch_app.kernels.functional_arithmetic_aliases",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    alias_out = alias_checker(*arithmetic_args)
    alias_eager = _eager_functional_arithmetic_aliases(*arithmetic_snap)
    assert _tensor_equal(alias_out, alias_eager)
    assert alias_out.device.type == "cpu"
    assert alias_out.dtype == torch.float32
    assert tuple(alias_out.shape) == (2, 3)
    assert alias_out.requires_grad is False
    for actual, snapshot in zip(arithmetic_args, arithmetic_snap, strict=True):
        assert _tensor_equal(actual, snapshot)

    matrix = torch.tensor(
        [[1.0, -2.0, 0.5], [3.0, 4.0, -1.0], [2.0, 0.25, 5.0]],
        dtype=torch.float32,
    )
    left_vector = torch.tensor([0.5, -1.5, 2.0], dtype=torch.float32)
    right_vector = torch.tensor([-2.0, 1.25, 0.75], dtype=torch.float32)
    mixed_args = (matrix, left_vector, right_vector)
    mixed_snap = _copy_tensor_args(mixed_args)
    mixed_checker = project.equivalence_checker(
        "torch_app.kernels.mixed_rank_matmul",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    mixed_out = mixed_checker(*mixed_args)
    mixed_eager = _eager_mixed_rank_matmul(*mixed_snap)
    assert _tensor_equal(mixed_out, mixed_eager)
    assert mixed_out.device.type == "cpu"
    assert mixed_out.dtype == torch.float32
    assert tuple(mixed_out.shape) == (3,)
    assert mixed_out.requires_grad is False
    for actual, snapshot in zip(mixed_args, mixed_snap, strict=True):
        assert _tensor_equal(actual, snapshot)

    unary_input = torch.tensor(
        [
            float("-inf"),
            -3.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            3.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    for qualname, _rule_id in unary_contracts:
        unary_snapshot = unary_input.detach().clone()
        unary_checker = project.equivalence_checker(
            f"torch_app.kernels.{qualname}",
            equals=_tensor_equal,
            args_equals=_args_unmutated,
            copy_args=_copy_tensor_args,
        )
        unary_out = unary_checker(unary_input)
        unary_eager = _eager_unary(qualname, unary_snapshot)
        assert _tensor_equal(unary_out, unary_eager)
        _assert_special_value_equivalence(unary_out, unary_eager)
        assert unary_out.device.type == "cpu"
        assert unary_out.dtype == torch.float32
        assert tuple(unary_out.shape) == (9,)
        assert unary_out.requires_grad is False
        assert _tensor_equal(unary_input, unary_snapshot)

    with _native_mode(project, "native"):
        from torch_app import kernels as native_kernels

        for qualname, _rule_id in unary_contracts:
            grad_input = unary_input.detach().clone().requires_grad_(True)
            unary_grad_out = getattr(native_kernels, qualname)(grad_input)
            unary_eager = _eager_unary(qualname, unary_input)
            _assert_special_value_equivalence(unary_grad_out, unary_eager)
            assert unary_grad_out.requires_grad is False
            assert _tensor_equal(grad_input.detach(), unary_input)

    for qualname, explicit_none in gelu_contracts:
        gelu_snapshot = unary_input.detach().clone()
        gelu_checker = project.equivalence_checker(
            f"torch_app.kernels.{qualname}",
            equals=_tensor_equal,
            args_equals=_args_unmutated,
            copy_args=_copy_tensor_args,
        )
        gelu_out = gelu_checker(unary_input)
        gelu_eager = _eager_gelu(gelu_snapshot, explicit_none=explicit_none)
        assert _tensor_equal(gelu_out, gelu_eager)
        _assert_special_value_equivalence(gelu_out, gelu_eager)
        assert gelu_out.device.type == "cpu"
        assert gelu_out.dtype == torch.float32
        assert tuple(gelu_out.shape) == (9,)
        assert gelu_out.requires_grad is False
        assert _tensor_equal(unary_input, gelu_snapshot)

    with _native_mode(project, "native"):
        from torch_app import kernels as native_kernels

        for qualname, explicit_none in gelu_contracts:
            grad_input = unary_input.detach().clone().requires_grad_(True)
            gelu_grad_out = getattr(native_kernels, qualname)(grad_input)
            gelu_eager = _eager_gelu(unary_input, explicit_none=explicit_none)
            _assert_special_value_equivalence(gelu_grad_out, gelu_eager)
            assert gelu_grad_out.requires_grad is False
            assert _tensor_equal(grad_input.detach(), unary_input)

    with _native_mode(project, "native"):
        from torch_app.kernels import mixed_rank_matmul

        with pytest.raises(RuntimeError):
            mixed_rank_matmul(
                torch.ones((2, 3), dtype=torch.float32),
                torch.ones(2, dtype=torch.float32),
                torch.ones(4, dtype=torch.float32),
            )

    with _native_mode(project, "native"):
        from torch_app.kernels import arithmetic_followup

        with pytest.raises(RuntimeError):
            arithmetic_followup(
                torch.ones((2, 2)),
                torch.ones((2, 2)),
                torch.ones(3),
                torch.ones(3),
            )

    logits = torch.tensor(
        [[-2.0, 0.5, 3.0], [1.0, -4.0, 2.0]], dtype=torch.float32
    )
    functional_checker = project.equivalence_checker(
        "torch_app.kernels.functional_classify",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    functional_out = functional_checker(logits)
    assert _tensor_equal(functional_out, _eager_functional_classify(logits))
    assert functional_out.dtype == torch.int64
    assert tuple(functional_out.shape) == (2,)
    assert functional_out.requires_grad is False

    vector_logits = torch.tensor([-3.0, 4.0, 1.0], dtype=torch.float32)
    vector_checker = project.equivalence_checker(
        "torch_app.kernels.vector_classify",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    vector_out = vector_checker(vector_logits)
    assert _tensor_equal(vector_out, _eager_vector_classify(vector_logits))
    assert vector_out.dtype == torch.int64
    assert tuple(vector_out.shape) == (1,)
    assert vector_out.requires_grad is False


class _native_mode:
    """Temporarily force REXTIO_NATIVE_MODE and load the generated package."""

    def __init__(self, project: CertifiedProject, mode: str) -> None:
        self.project = project
        self.mode = mode
        self._previous: str | None = None
        self._inserted = False
        self._build_dir = str(project.build_python_dir)

    def __enter__(self) -> None:
        _import_torch()
        self._previous = os.environ.get("REXTIO_NATIVE_MODE")
        os.environ["REXTIO_NATIVE_MODE"] = self.mode
        for name in list(sys.modules):
            if (
                name == "_rextio_native"
                or name == "torch_app"
                or name.startswith("torch_app.")
            ):
                sys.modules.pop(name, None)
        if self._build_dir not in sys.path:
            sys.path.insert(0, self._build_dir)
            self._inserted = True

    def __exit__(self, *exc: object) -> None:
        if self._inserted and self._build_dir in sys.path:
            sys.path.remove(self._build_dir)
        for name in list(sys.modules):
            if (
                name == "_rextio_native"
                or name == "torch_app"
                or name.startswith("torch_app.")
            ):
                sys.modules.pop(name, None)
        if self._previous is None:
            os.environ.pop("REXTIO_NATIVE_MODE", None)
        else:
            os.environ["REXTIO_NATIVE_MODE"] = self._previous
