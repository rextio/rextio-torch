"""Focused contracts for the diagnostic small-batch scoring harness."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import bench_small_batch_scoring as bench
from benchmarks.small_batch_scoring_cases import (
    BATCH_SIZES,
    CLASS_COUNT,
    CONTROL_FLOW_ROUNDS,
    DEFAULT_TIMING_SAMPLES,
    EXPECTED_NATIVE_ROUTE,
    FEATURE_WIDTH,
    FROZEN_CASES,
    KERNEL_SOURCE,
    SCORE_LABELS_QUALNAME,
    SCORE_LOGITS_QUALNAME,
    SCORE_PROBS_QUALNAME,
    frozen_cases,
    get_case,
)
from rextio_torch.invocation_scope import INVOCATION_SCOPE_OPTIMIZATION_ACTIVE


def _write_accepted_check_json(project: Path) -> None:
    reports = project / ".rextio" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "check.json").write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "functions": [
                            {
                                "qualname": name,
                                "native_status": "accepted",
                                "route": EXPECTED_NATIVE_ROUTE,
                            }
                            for name in (
                                SCORE_LOGITS_QUALNAME,
                                SCORE_PROBS_QUALNAME,
                                SCORE_LABELS_QUALNAME,
                            )
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_provenance_faithful_build(
    project: Path,
    *,
    kernels_body: str | None = None,
    native_body: str | None = None,
    place_native_in_build: bool = True,
) -> Path:
    """Create a build tree with torch_score + _rextio_native under the build root."""
    build_python = project / ".rextio" / "build" / "python"
    package = build_python / "torch_score"
    package.mkdir(parents=True, exist_ok=True)
    _write_accepted_check_json(project)
    if place_native_in_build:
        (build_python / "_rextio_native.py").write_text(
            native_body
            if native_body is not None
            else "# provenance-faithful fake native extension\n",
            encoding="utf-8",
        )
    else:
        outside = project / "outside_native"
        outside.mkdir(parents=True, exist_ok=True)
        (outside / "_rextio_native.py").write_text(
            "# wrong-tree fake native extension\n",
            encoding="utf-8",
        )
    (package / "__init__.py").write_text("", encoding="utf-8")
    default_kernels = (
        "import _rextio_native  # noqa: F401\n"
        "\n"
        "def score_logits(*args):\n"
        "    return 'logits'\n"
        "\n"
        "def score_probabilities(*args):\n"
        "    return 'probs'\n"
        "\n"
        "def score_labels(*args):\n"
        "    return 'labels'\n"
    )
    (package / "kernels.py").write_text(
        kernels_body if kernels_body is not None else default_kernels,
        encoding="utf-8",
    )
    return build_python


def test_predeclared_matrix_uses_control_flow_rounds_four() -> None:
    assert BATCH_SIZES == (1, 16, 128)
    assert FEATURE_WIDTH == 32
    assert CLASS_COUNT == 8
    assert CONTROL_FLOW_ROUNDS == 4
    assert DEFAULT_TIMING_SAMPLES >= 1
    assert [case.case_id for case in FROZEN_CASES] == ["s1", "s16", "s128"]
    assert [case.batch for case in frozen_cases()] == [1, 16, 128]
    for case in FROZEN_CASES:
        assert case.feature_width == 32
        assert case.hidden_width == 32
        assert case.class_count == 8
        assert case.rounds == 4
        assert case.phase == 0
        assert case.logits_shape == (case.batch, 8)
        assert case.labels_shape == (case.batch,)
        assert not hasattr(case, "depth")
    assert get_case("s16") is FROZEN_CASES[1]
    with pytest.raises(KeyError):
        get_case("missing")


def test_kernel_source_covers_required_pipeline_ops() -> None:
    assert "x - mean" in KERNEL_SOURCE
    assert "centered / scale" in KERNEL_SOURCE
    assert KERNEL_SOURCE.count("F.linear(") == 6  # three entry points × two linears
    assert "for layer in range(rounds):" in KERNEL_SOURCE
    assert "for layer in range(depth):" not in KERNEL_SOURCE
    assert "rounds: int" in KERNEL_SOURCE
    assert "if (layer + phase) % 2 == 0:" in KERNEL_SOURCE
    assert ".relu()" in KERNEL_SOURCE
    assert ".tanh()" in KERNEL_SOURCE
    assert ".softmax(dim=1)" in KERNEL_SOURCE
    assert ".argmax(dim=1, keepdim=False)" in KERNEL_SOURCE
    assert "def score_logits" in KERNEL_SOURCE
    assert "def score_probabilities" in KERNEL_SOURCE
    assert "def score_labels" in KERNEL_SOURCE


def test_timing_samples_positive_validated() -> None:
    assert bench.require_positive_timing_samples(1) == 1
    assert bench.require_positive_timing_samples(4) == 4
    with pytest.raises(RuntimeError, match="timing_samples must be >= 1"):
        bench.require_positive_timing_samples(0)
    with pytest.raises(RuntimeError, match="positive int"):
        bench.require_positive_timing_samples(True)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="positive int"):
        bench.require_positive_timing_samples(1.5)  # type: ignore[arg-type]


def test_refuse_historical_phase_outputs(tmp_path: Path) -> None:
    assert bench._is_historical_authoritative_path(bench.RESULTS_PHASE_A / "latest.json")
    assert bench._is_historical_authoritative_path(
        bench.RESULTS_PHASE_B / "evidence" / "check.json"
    )
    assert not bench._is_historical_authoritative_path(tmp_path / "diag.json")
    with pytest.raises(RuntimeError, match="historical benchmark tree"):
        bench.refuse_historical_output(bench.RESULTS_PHASE_A / "latest.json")
    with pytest.raises(RuntimeError, match="historical benchmark tree"):
        bench.write_diagnostic_output({"ok": True}, bench.RESULTS_PHASE_B / "report.md")


def test_build_report_is_diagnostic_only_and_scope_inactive() -> None:
    report = bench.build_report([])
    assert report["timing_class"] == "diagnostic"
    assert report["official_cohort"] is False
    assert report["speedup_claim"] is None
    assert report["control_flow_rounds"] == 4
    assert report["timing_samples"] == DEFAULT_TIMING_SAMPLES
    assert report["native_measured"] is False
    assert report["native_lane"]["available"] is False
    assert report["native_lane"]["measured"] is False
    assert "built-project" in report["native_lane"]["reason"]
    assert report["core_function_scope"]["invocation_scope_optimization_active"] is False
    assert report["core_function_scope"]["has_function_body_scope_hook"] is False
    assert INVOCATION_SCOPE_OPTIMIZATION_ACTIVE is False
    assert "per-operation" in report["production_no_grad_baseline"] or (
        "each rextio-torch tch helper" in report["production_no_grad_baseline"]
    )


def test_native_omission_is_explicit_when_lane_absent() -> None:
    torch = pytest.importorskip("torch")
    case = get_case("s1")
    cell = bench.run_case_diagnostic(torch, case, timing_samples=1, native_lane=None)
    assert cell["native_lane_available"] is False
    assert cell["native_measured"] is False
    native_timing = cell["timings"]["native_rextio"]
    assert native_timing["available"] is False
    assert native_timing["measured"] is False
    assert native_timing["speedup_claim"] is None
    assert "built-project" in native_timing["reason"]
    assert cell["correctness"]["native_rextio"]["measured"] is False
    assert cell["timings"]["eager"]["measured"] is True
    assert cell["timings"]["eager"]["role"] == "context"
    assert "torch_inference_mode" in cell["timings"]


def test_native_inclusion_uses_synthetic_lane_without_process_mutation() -> None:
    torch = pytest.importorskip("torch")
    case = get_case("s1")
    tensors = bench.make_case_tensors(torch, case)
    expected_logits = bench.eager_score_logits(torch, **tensors)
    expected_probs = bench.eager_score_probabilities(torch, **tensors)
    expected_labels = bench.eager_score_labels(torch, **tensors)

    lane = bench.make_synthetic_native_lane(
        score_logits=lambda *args: expected_logits,
        score_probabilities=lambda *args: expected_probs,
        score_labels=lambda *args: expected_labels,
        native_extension_path="/synthetic/build/_rextio_native.so",
    )
    assert lane.process_state_owned is False
    cell = bench.run_case_diagnostic(torch, case, timing_samples=1, native_lane=lane)
    assert cell["native_lane_available"] is True
    assert cell["native_measured"] is True
    assert cell["control_flow_rounds"] == 4
    assert cell["correctness"]["native_rextio"]["logits_ok"] is True
    assert cell["correctness"]["native_rextio"]["probabilities_ok"] is True
    assert cell["correctness"]["native_rextio"]["labels_ok"] is True
    native_timing = cell["timings"]["native_rextio"]
    assert native_timing["available"] is True
    assert native_timing["measured"] is True
    assert native_timing["rebuild_inside_timing"] is False
    assert native_timing["speedup_claim"] is None
    assert native_timing["role"] == "primary_diagnostic"
    report = bench.build_report([cell], timing_samples=1, native_lane=lane)
    assert report["native_measured"] is True
    assert report["native_lane"]["native_extension_path"] == (
        "/synthetic/build/_rextio_native.so"
    )
    lane.close()
    lane.close()  # idempotent
    assert lane.closed is True


def test_verify_native_routes_fail_closed(tmp_path: Path) -> None:
    reports = tmp_path / ".rextio" / "reports"
    reports.mkdir(parents=True)
    (reports / "check.json").write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "functions": [
                            {
                                "qualname": SCORE_LOGITS_QUALNAME,
                                "native_status": "accepted",
                                "route": EXPECTED_NATIVE_ROUTE,
                            },
                            {
                                "qualname": SCORE_PROBS_QUALNAME,
                                "native_status": "accepted",
                                "route": EXPECTED_NATIVE_ROUTE,
                            },
                            {
                                "qualname": SCORE_LABELS_QUALNAME,
                                "native_status": "rejected",
                                "route": "fallback-python",
                            },
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not native-plugin accepted"):
        bench.verify_native_routes(tmp_path)


def test_load_native_lane_lifecycle_path_mode_and_extension_provenance(
    tmp_path: Path,
) -> None:
    project = tmp_path / "proj"
    build_python = _write_provenance_faithful_build(project)
    build_str = str(build_python.resolve())

    prior_mode = os.environ.get("REXTIO_NATIVE_MODE")
    os.environ["REXTIO_NATIVE_MODE"] = "fallback"
    # Path present later must still be forced to index 0.
    if build_str in sys.path:
        while build_str in sys.path:
            sys.path.remove(build_str)
    sys.path.append(build_str)
    prior_path = list(sys.path)
    try:
        with bench.load_native_scoring_lane(project) as lane:
            assert lane.available is True
            assert lane.closed is False
            assert lane.process_state_owned is True
            assert sys.path[0] == build_str
            assert sys.path.count(build_str) == 1
            assert os.environ.get("REXTIO_NATIVE_MODE") == "native"
            assert lane.score_logits() == "logits"
            assert lane.score_probabilities() == "probs"
            assert lane.score_labels() == "labels"
            ext_path = Path(lane.native_extension_path).resolve()
            assert ext_path.is_relative_to(build_python.resolve())
            assert bench.NATIVE_EXTENSION_MODULE in sys.modules
            assert "torch_score" in sys.modules or "torch_score.kernels" in sys.modules
            report = lane.to_report_dict()
            assert report["native_extension_path"] == lane.native_extension_path
            assert report["native_extension_module"] == "_rextio_native"
            assert report["rebuild_inside_timing"] is False

        assert lane.closed is True
        assert os.environ.get("REXTIO_NATIVE_MODE") == "fallback"
        assert list(sys.path) == prior_path
        assert "torch_score" not in sys.modules
        assert "torch_score.kernels" not in sys.modules
        assert bench.NATIVE_EXTENSION_MODULE not in sys.modules

        # Idempotent close.
        lane.close()
        assert lane.closed is True
        assert os.environ.get("REXTIO_NATIVE_MODE") == "fallback"

        # CertifiedProject-shaped object.
        certified = SimpleNamespace(
            project_root=project,
            build_python_dir=build_python,
        )
        with bench.load_native_scoring_lane(certified) as lane2:
            assert lane2.score_labels() == "labels"
            assert Path(lane2.native_extension_path).resolve().is_relative_to(
                build_python.resolve()
            )
        assert lane2.closed is True
        assert bench.NATIVE_EXTENSION_MODULE not in sys.modules
    finally:
        if prior_mode is None:
            os.environ.pop("REXTIO_NATIVE_MODE", None)
        else:
            os.environ["REXTIO_NATIVE_MODE"] = prior_mode
        while build_str in sys.path:
            sys.path.remove(build_str)
        bench._evict_lane_modules("torch_score")


def test_load_native_lane_path_front_failure_restores_after_partial_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup must run even when path-front fails mid-mutation after snapshots."""
    project = tmp_path / "proj_path_front_fail"
    build_python = _write_provenance_faithful_build(project)
    build_str = str(build_python.resolve())

    prior_mode_present = "REXTIO_NATIVE_MODE" in os.environ
    prior_mode = os.environ.get("REXTIO_NATIVE_MODE")
    os.environ.pop("REXTIO_NATIVE_MODE", None)
    # Ensure a polluted module entry is cleared by failure cleanup.
    sys.modules["torch_score"] = type(sys)("torch_score")  # type: ignore[assignment]
    sys.modules["_rextio_native"] = type(sys)("_rextio_native")  # type: ignore[assignment]
    prior_path = list(sys.path)

    def boom_after_mutating_path(build_dir_str: str) -> None:
        sys.path[:] = [entry for entry in sys.path if entry != build_dir_str]
        sys.path.insert(0, build_dir_str)
        sys.path.append("/tmp/partial-path-front-side-effect")
        raise RuntimeError("path-front boom")

    monkeypatch.setattr(bench, "_force_build_dir_to_sys_path_front", boom_after_mutating_path)
    try:
        with pytest.raises(RuntimeError, match="path-front boom"):
            bench.load_native_scoring_lane(project)
        assert "REXTIO_NATIVE_MODE" not in os.environ
        assert list(sys.path) == prior_path
        assert "torch_score" not in sys.modules
        assert "_rextio_native" not in sys.modules
        assert build_str not in sys.path or build_str in prior_path
        assert "/tmp/partial-path-front-side-effect" not in sys.path
    finally:
        if prior_mode_present:
            assert prior_mode is not None
            os.environ["REXTIO_NATIVE_MODE"] = prior_mode
        else:
            os.environ.pop("REXTIO_NATIVE_MODE", None)
        bench._evict_lane_modules("torch_score")
        # Defensive: restore path if a regression left residue.
        sys.path[:] = prior_path


def test_load_native_lane_failure_restores_env_path_and_partial_modules(
    tmp_path: Path,
) -> None:
    project = tmp_path / "proj_fail"
    build_python = _write_provenance_faithful_build(
        project,
        kernels_body="import _rextio_native\n\nraise RuntimeError('import boom')\n",
    )
    build_str = str(build_python.resolve())
    prior_mode_present = "REXTIO_NATIVE_MODE" in os.environ
    prior_mode = os.environ.get("REXTIO_NATIVE_MODE")
    os.environ.pop("REXTIO_NATIVE_MODE", None)
    prior_path = list(sys.path)
    try:
        with pytest.raises(RuntimeError, match="import boom"):
            bench.load_native_scoring_lane(project)
        assert "REXTIO_NATIVE_MODE" not in os.environ
        assert list(sys.path) == prior_path
        assert "torch_score" not in sys.modules
        assert "torch_score.kernels" not in sys.modules
        assert bench.NATIVE_EXTENSION_MODULE not in sys.modules
        # Even if build path was injected mid-failure, it must not stick at front.
        if build_str in sys.path:
            assert sys.path[0] != build_str or build_str in prior_path
    finally:
        if prior_mode_present:
            assert prior_mode is not None
            os.environ["REXTIO_NATIVE_MODE"] = prior_mode
        else:
            os.environ.pop("REXTIO_NATIVE_MODE", None)
        bench._evict_lane_modules("torch_score")


def test_load_native_lane_rejects_wrong_extension_tree(tmp_path: Path) -> None:
    project = tmp_path / "proj_wrong_ext"
    build_python = _write_provenance_faithful_build(
        project,
        place_native_in_build=False,
        kernels_body=(
            "import sys\n"
            "from pathlib import Path\n"
            # kernels.py → torch_score → python → build → .rextio → project
            "outside = Path(__file__).resolve().parents[4] / 'outside_native'\n"
            "sys.path.insert(0, str(outside))\n"
            "import _rextio_native  # noqa: F401\n"
            "\n"
            "def score_logits(*args):\n"
            "    return 'logits'\n"
            "\n"
            "def score_probabilities(*args):\n"
            "    return 'probs'\n"
            "\n"
            "def score_labels(*args):\n"
            "    return 'labels'\n"
        ),
    )
    prior_path = list(sys.path)
    prior_mode_present = "REXTIO_NATIVE_MODE" in os.environ
    prior_mode = os.environ.get("REXTIO_NATIVE_MODE")
    try:
        with pytest.raises(RuntimeError, match="_rextio_native"):
            bench.load_native_scoring_lane(project)
        assert list(sys.path) == prior_path
        if prior_mode_present:
            assert os.environ.get("REXTIO_NATIVE_MODE") == prior_mode
        else:
            assert "REXTIO_NATIVE_MODE" not in os.environ
        assert bench.NATIVE_EXTENSION_MODULE not in sys.modules
        assert "torch_score.kernels" not in sys.modules
    finally:
        if prior_mode_present:
            assert prior_mode is not None
            os.environ["REXTIO_NATIVE_MODE"] = prior_mode
        else:
            os.environ.pop("REXTIO_NATIVE_MODE", None)
        bench._evict_lane_modules("torch_score")
        assert build_python.is_dir()


def test_closed_lane_rejected_by_diagnostic() -> None:
    torch = pytest.importorskip("torch")
    case = get_case("s1")
    tensors = bench.make_case_tensors(torch, case)
    labels = bench.eager_score_labels(torch, **tensors)
    lane = bench.make_synthetic_native_lane(
        score_logits=lambda *a: bench.eager_score_logits(torch, **tensors),
        score_probabilities=lambda *a: bench.eager_score_probabilities(torch, **tensors),
        score_labels=lambda *a: labels,
    )
    lane.close()
    with pytest.raises(RuntimeError, match="closed"):
        bench.run_case_diagnostic(torch, case, timing_samples=1, native_lane=lane)


def test_validate_pipeline_order_rejects_label_only_match() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=1)
    labels = probs.argmax(dim=1, keepdim=False)
    drifted = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
    drifted_probs = torch.softmax(drifted, dim=1)
    with pytest.raises(RuntimeError, match="full logits diverged"):
        bench.validate_pipeline_outputs(
            torch,
            logits=drifted,
            probabilities=drifted_probs,
            labels=labels,
            expected_logits=logits,
            expected_probabilities=probs,
            expected_labels=labels,
        )


def test_eager_pipeline_and_diagnostic_case_run() -> None:
    torch = pytest.importorskip("torch")
    case = get_case("s1")
    tensors = bench.make_case_tensors(torch, case)
    assert tensors["rounds"] == 4
    logits = bench.eager_score_logits(torch, **tensors)
    probs = bench.eager_score_probabilities(torch, **tensors)
    labels = bench.eager_score_labels(torch, **tensors)
    assert tuple(logits.shape) == case.logits_shape
    assert tuple(probs.shape) == case.logits_shape
    assert tuple(labels.shape) == case.labels_shape
    assert labels.dtype == torch.int64
    cell = bench.run_case_diagnostic(torch, case, timing_samples=1)
    assert cell["timing_class"] == "diagnostic"
    assert cell["official_cohort"] is False
    assert cell["speedup_claim"] is None
    assert cell["control_flow_rounds"] == 4
    assert cell["timing_samples"] == 1
    assert cell["correctness"]["eager"]["logits_ok"] is True
    assert cell["correctness"]["eager"]["probabilities_ok"] is True
    assert cell["correctness"]["eager"]["labels_ok"] is True
    assert cell["correctness"]["eager"]["validation_order"] == [
        "logits",
        "probabilities",
        "labels",
    ]
    assert cell["timings"]["eager"]["timing_class"] == "diagnostic"
    assert cell["timings"]["eager"]["speedup_claim"] is None
    assert cell["timings"]["eager"]["timing_samples"] == 1
    assert cell["invocation_scope_optimization_active"] is False
    assert cell["timings"]["native_rextio"]["measured"] is False
    if cell["timings"]["torch_inference_mode"].get("measured"):
        assert cell["timings"]["torch_inference_mode"]["role"] == "context"


def test_write_diagnostic_output_to_temp(tmp_path: Path) -> None:
    path = tmp_path / "small-batch-scoring.json"
    report = bench.build_report([])
    bench.write_diagnostic_output(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["timing_class"] == "diagnostic"
    assert loaded["official_cohort"] is False
    assert loaded["control_flow_rounds"] == 4
