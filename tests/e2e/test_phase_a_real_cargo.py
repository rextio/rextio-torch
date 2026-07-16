"""One real-Cargo E2E certification for the Phase A linear→ReLU→mean chain.

Uses the core certification kit (``rextio.plugins.testing``) with the
entry-point-discoverable plugin. Build env requires the local CPython 3.11
venv's torch 2.11.0 via ``LIBTORCH_USE_PYTORCH=1`` (never
``LIBTORCH_BYPASS_VERSION_CHECK``). Torch is imported before the generated
native extension loads.
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

KERNELS = """
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch.nn.functional as F


def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
"""


def _configure_libtorch_env() -> None:
    """Point the tch/PyO3 build at this project's CPython 3.11 + torch 2.11 venv.

    ``torch-sys`` 0.24 with ``LIBTORCH_USE_PYTORCH=1`` shells out to bare
    ``python`` (when ``VIRTUAL_ENV`` is set) or ``python3`` otherwise — it does
    **not** honor ``PYO3_PYTHON``. Put the dedicated venv's ``bin`` first on
    ``PATH`` and set ``VIRTUAL_ENV`` so that interpreter is the one that reports
    torch 2.11.0. ``PYO3_PYTHON`` is still set for PyO3's extension ABI pin.
    """
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
    # Fail closed if PATH still resolves a mismatched torch for torch-sys.
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
    """Import torch before any native extension load (tch python-extension)."""
    import torch

    return torch


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> CertifiedProject:
    _configure_libtorch_env()
    # Import torch in this process before the native extension is ever loaded.
    _import_torch()

    root = tmp_path_factory.mktemp("torch_phase_a")
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
    return _tensor_equal(left, right)


def _copy_tensor_args(args: tuple[object, ...]) -> tuple[object, ...]:
    torch = _import_torch()
    copies: list[object] = []
    for arg in args:
        if type(arg) is torch.Tensor:
            copies.append(arg.detach().clone())
        else:
            copies.append(arg)
    return tuple(copies)


def test_phase_a_chain_real_cargo_certification(project: CertifiedProject) -> None:
    """Single serialized build: route evidence, native≈eager, contracts, lifetime."""
    torch = _import_torch()
    assert torch.__version__.startswith("2.11.")
    assert sys.version_info[:2] == (3, 11)

    # --- build/check route evidence ---
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
    assert len(claims) == 3
    claim_rules = {claim["rule_id"] for claim in claims}
    assert claim_rules == {
        "rextio-torch/functional-linear-f32-cpu-2d",
        "rextio-torch/tensor-relu-f32-cpu-2d",
        "rextio-torch/tensor-mean-dim1-f32-cpu-2d",
    }
    by_rule = {claim["rule_id"]: claim for claim in claims}
    assert by_rule["rextio-torch/functional-linear-f32-cpu-2d"]["result_type"] == (
        "rextio-torch/tensor-f32-cpu-2d"
    )
    assert by_rule["rextio-torch/tensor-relu-f32-cpu-2d"]["result_type"] == (
        "rextio-torch/tensor-f32-cpu-2d"
    )
    assert by_rule["rextio-torch/tensor-mean-dim1-f32-cpu-2d"]["result_type"] == (
        "rextio-torch/tensor-f32-cpu-1d"
    )

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "struct RxtTorchTensor" in rust
    assert "__rxttorch_linear" in rust
    assert "__rxttorch_relu" in rust
    assert "__rxttorch_mean_dim1_keepdim_false" in rust
    assert "no_grad_guard" in rust
    assert "f_linear" in rust
    assert "f_relu" in rust
    assert "f_mean_dim" in rust
    assert "LIBTORCH_BYPASS_VERSION_CHECK" not in os.environ

    # Fixed small CPU float32 fixtures (N=4, in=3, out=2).
    torch.manual_seed(0)
    x = torch.randn(4, 3, dtype=torch.float32)
    weight = torch.randn(2, 3, dtype=torch.float32)
    bias = torch.randn(2, dtype=torch.float32)
    x_snap = x.detach().clone()
    w_snap = weight.detach().clone()
    b_snap = bias.detach().clone()

    checker = project.equivalence_checker(
        "torch_app.kernels.inference",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    native_out = checker(x, weight, bias)

    # --- native-mode output shape / dtype / device / no-grad ---
    assert type(native_out) is torch.Tensor
    assert native_out.device.type == "cpu"
    assert native_out.dtype == torch.float32
    assert native_out.dim() == 1
    assert tuple(native_out.shape) == (4,)
    assert native_out.requires_grad is False

    # Eager reference (same graph as the annotated fallback body).
    eager = torch.nn.functional.linear(x_snap, w_snap, b_snap).relu().mean(dim=1, keepdim=False)
    assert _tensor_equal(native_out, eager)

    # Inputs not mutated by the certified call path (post-call tensors still match snaps).
    assert torch.equal(x, x_snap)
    assert torch.equal(weight, w_snap)
    assert torch.equal(bias, b_snap)

    # --- requires_grad inputs still yield a no-grad native output ---
    # Eager fallback can attach an autograd graph when inputs request gradients;
    # native helpers always run under no_grad, so this leg is native-only.
    x_g = x_snap.detach().clone().requires_grad_(True)
    w_g = w_snap.detach().clone().requires_grad_(True)
    b_g = b_snap.detach().clone().requires_grad_(True)
    with _native_mode(project, "native"):
        from torch_app.kernels import inference as inference_fn

        grad_out = inference_fn(x_g, w_g, b_g)
    assert type(grad_out) is torch.Tensor
    assert grad_out.requires_grad is False
    assert grad_out.device.type == "cpu"
    assert grad_out.dtype == torch.float32
    assert grad_out.dim() == 1
    assert torch.allclose(grad_out, eager, rtol=1e-5, atol=1e-6)
    # Grad-requesting inputs remain non-mutated (values; leaf storage intact).
    assert torch.equal(x_g.detach(), x_snap)
    assert torch.equal(w_g.detach(), w_snap)
    assert torch.equal(b_g.detach(), b_snap)

    # --- returned output lifetime after inputs are released ---
    x_live = x_snap.detach().clone()
    w_live = w_snap.detach().clone()
    b_live = b_snap.detach().clone()
    with _native_mode(project, "native"):
        from torch_app.kernels import inference as inference_live

        held = inference_live(x_live, w_live, b_live)
    del x_live, w_live, b_live
    import gc

    gc.collect()
    assert type(held) is torch.Tensor
    assert held.device.type == "cpu"
    assert held.dtype == torch.float32
    assert held.dim() == 1
    assert torch.allclose(held, eager, rtol=1e-5, atol=1e-6)

    # --- fail-closed boundary on annotation-violating runtime values ---
    # Same serialized build; native-only so a successful eager fallback cannot
    # mask a boundary miss (equivalence would diverge).
    w_ok = w_snap.detach().clone()
    b_ok = b_snap.detach().clone()
    with _native_mode(project, "native"):
        from torch_app.kernels import inference as inference_boundary

        # float64 where float32 is promised (x: TensorF32Cpu2D).
        x_f64 = torch.randn(4, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"^rextio-torch: expected a float32 tensor$") as dtype_info:
            inference_boundary(x_f64, w_ok, b_ok)
        assert str(dtype_info.value) == RUNTIME_ERRORS["dtype"]

        # rank-1 where rank-2 is promised (x: TensorF32Cpu2D).
        x_rank1 = torch.randn(3, dtype=torch.float32)
        with pytest.raises(
            ValueError,
            match=r"^rextio-torch: expected rank-2 tensor, got rank 1$",
        ) as rank_info:
            inference_boundary(x_rank1, w_ok, b_ok)
        assert str(rank_info.value) == "rextio-torch: expected rank-2 tensor, got rank 1"

    # Valid weights/bias still intact after the rejected calls (no crash/mutation).
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
        # Torch must already be imported in this process.
        _import_torch()
        self._previous = os.environ.get("REXTIO_NATIVE_MODE")
        os.environ["REXTIO_NATIVE_MODE"] = self.mode
        # Evict previous generated modules.
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
