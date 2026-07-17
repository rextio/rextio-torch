"""Real-Cargo vertical slice: Python for/if around Alpha AOT tch ops.

Certifies that scalar runtime control flow lowers to Rust control flow around
native tch helpers. A second function in the same serialized Cargo project
exercises functional/method matmul, same-rank add, sum, tanh, and rank-1
activation chaining, without paying for another build. Evidence includes native
routes, numeric equivalence, dtype/device/rank, no-grad, non-mutation, and
fail-closed boundary rejects.

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
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
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
