"""Opt-in real-Cargo vertical slice for the diagnostic scoring pipeline.

Builds a dedicated certification project (not the Alpha mega-surface) so the
0.1.3 diagnostic scoring path remains maintainable. Skips explicitly when cargo
or the dedicated CPython 3.11 + torch 2.11 venv is unavailable.

Exercises the harness native diagnostic lane for all three predeclared cells
via retained wrappers (no rebuild inside timed samples).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rextio.plugins.testing import CertifiedProject, build_certification_project

from benchmarks.bench_small_batch_scoring import (
    load_native_scoring_lane,
    run_diagnostic_matrix,
)
from benchmarks.small_batch_scoring_cases import (
    BATCH_SIZES,
    CONTROL_FLOW_ROUNDS,
    FROZEN_CASES,
    KERNEL_SOURCE,
    SCORE_LABELS_QUALNAME,
    SCORE_LOGITS_QUALNAME,
    SCORE_PROBS_QUALNAME,
)
from rextio_torch.invocation_scope import INVOCATION_SCOPE_OPTIMIZATION_ACTIVE

pytestmark = [
    pytest.mark.needs_cargo,
    pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="real-Cargo small-batch scoring requires cargo on PATH",
    ),
]

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PLUGIN_ROOT / ".venv" / "bin" / "python"


def _configure_libtorch_env() -> None:
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
    probe = subprocess.run(
        ["python", "-c", "import torch; print(torch.__version__.split('+')[0])"],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    if probe.returncode != 0:
        pytest.skip(f"torch probe failed: {probe.stderr}")
    version = probe.stdout.strip()
    if version != "2.11.0":
        pytest.skip(
            f"torch-sys build python must be torch 2.11.0; PATH python reports {version!r}"
        )


def _import_torch():
    import torch

    return torch


def _route_of(project: CertifiedProject, qualname: str) -> dict:
    report = json.loads(
        (project.project_root / ".rextio" / "reports" / "check.json").read_text(
            encoding="utf-8"
        )
    )
    for module in report["modules"]:
        for function in module["functions"]:
            if function["qualname"] == qualname:
                return function
    raise AssertionError(f"{qualname} not found in check.json")


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> CertifiedProject:
    _configure_libtorch_env()
    _import_torch()
    root = tmp_path_factory.mktemp("torch_small_batch_scoring")
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-torch"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "torch_score"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNEL_SOURCE, encoding="utf-8")
    return build_certification_project(root)


def test_small_batch_scoring_real_cargo(project: CertifiedProject) -> None:
    """Native route, per-op no_grad baseline, diagnostic lane for all cells."""
    torch = _import_torch()
    if not torch.__version__.startswith("2.11."):
        pytest.skip(f"expected torch 2.11.x, got {torch.__version__}")
    if sys.version_info[:2] != (3, 11):
        pytest.skip(f"expected CPython 3.11, got {sys.version_info[:2]}")

    build = json.loads(
        (project.project_root / ".rextio" / "reports" / "build.json").read_text(
            encoding="utf-8"
        )
    )
    assert build["status"] == "built"

    for qualname in (
        SCORE_LOGITS_QUALNAME,
        SCORE_PROBS_QUALNAME,
        SCORE_LABELS_QUALNAME,
    ):
        record = _route_of(project, qualname)
        assert record["native_status"] == "accepted"
        assert record["route"] == "native-plugin:rextio-torch"

    rust = (
        project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs"
    ).read_text(encoding="utf-8")
    assert "let __rextio_plugin_scope_guard_" in rust
    assert "tch::no_grad_guard()" in rust
    assert "_rxttorch_invocation_no_grad" not in rust
    assert INVOCATION_SCOPE_OPTIMIZATION_ACTIVE is True
    assert "for " in rust or "loop " in rust
    assert "if " in rust
    assert "__rxttorch_sub_function_scoped" in rust
    assert "__rxttorch_div_function_scoped" in rust
    assert "__rxttorch_linear_function_scoped" in rust
    assert "__rxttorch_relu_function_scoped" in rust
    assert "__rxttorch_tanh_function_scoped" in rust
    assert "__rxttorch_softmax_dim1_function_scoped" in rust
    assert "__rxttorch_argmax_dim1_keepdim_false_function_scoped" in rust

    prior_mode_present = "REXTIO_NATIVE_MODE" in os.environ
    prior_mode = os.environ.get("REXTIO_NATIVE_MODE")
    prior_path = list(sys.path)
    build_str = str(project.build_python_dir.resolve())

    # Import/retain native wrappers once outside timed samples; always close.
    with load_native_scoring_lane(project) as native_lane:
        assert native_lane.available is True
        assert native_lane.closed is False
        assert native_lane.build_python_dir == project.build_python_dir.resolve()
        assert native_lane.routes[SCORE_LABELS_QUALNAME]["route"] == (
            "native-plugin:rextio-torch"
        )
        ext_path = Path(native_lane.native_extension_path).resolve()
        assert ext_path.is_file() or ext_path.exists()
        assert ext_path.is_relative_to(project.build_python_dir.resolve())
        assert sys.path[0] == build_str
        assert os.environ.get("REXTIO_NATIVE_MODE") == "native"

        report = run_diagnostic_matrix(
            torch,
            timing_samples=1,
            native_lane=native_lane,
        )
        assert report["timing_class"] == "diagnostic"
        assert report["official_cohort"] is False
        assert report["speedup_claim"] is None
        assert report["control_flow_rounds"] == CONTROL_FLOW_ROUNDS == 4
        assert report["native_measured"] is True
        assert report["native_lane"]["available"] is True
        assert report["native_lane"]["rebuild_inside_timing"] is False
        assert report["native_lane"]["native_extension_path"] == (
            native_lane.native_extension_path
        )

        assert [cell["case"]["batch"] for cell in report["cells"]] == list(BATCH_SIZES)
        assert len(report["cells"]) == len(FROZEN_CASES) == 3
        for cell in report["cells"]:
            assert cell["case"]["rounds"] == 4
            assert cell["control_flow_rounds"] == 4
            assert cell["native_measured"] is True
            native_correctness = cell["correctness"]["native_rextio"]
            assert native_correctness["logits_ok"] is True
            assert native_correctness["probabilities_ok"] is True
            assert native_correctness["labels_ok"] is True
            assert native_correctness["validation_order"] == [
                "logits",
                "probabilities",
                "labels",
            ]
            native_timing = cell["timings"]["native_rextio"]
            assert native_timing["timing_class"] == "diagnostic"
            assert native_timing["available"] is True
            assert native_timing["measured"] is True
            assert native_timing["rebuild_inside_timing"] is False
            assert native_timing["speedup_claim"] is None
            assert native_timing["role"] == "primary_diagnostic"
            # Context lanes remain present.
            assert cell["timings"]["eager"]["role"] == "context"
            assert "torch_inference_mode" in cell["timings"]

    assert native_lane.closed is True
    assert list(sys.path) == prior_path
    if prior_mode_present:
        assert os.environ.get("REXTIO_NATIVE_MODE") == prior_mode
    else:
        assert "REXTIO_NATIVE_MODE" not in os.environ
    assert "torch_score" not in sys.modules
    assert "torch_score.kernels" not in sys.modules
    assert "_rextio_native" not in sys.modules
