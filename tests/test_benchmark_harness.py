"""Focused unit tests for the Phase A benchmark harness (no performance timing)."""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import pytest

from benchmarks import bench_phase_a as bench
from benchmarks.bench_phase_a import (
    FLOOR_NS,
    GO_AGGREGATE_CI_LOW_MIN,
    GO_AGGREGATE_GM_SPEEDUP_MIN,
    GO_MAX_CELL_MEDIAN_RATIO,
    _calibrate_common_iterations,
    _cell_primary_eligible,
    _collect_paired_samples_with_retry,
    _counterbalanced_schedule,
    _is_authoritative_path,
    _paired_log_bootstrap,
    _percentile,
    _require,
    _schedule_balance,
    _validate_core_provenance,
    _validate_plugin_direct_url,
    compute_expansion_go_gate,
)
from benchmarks.cases import (
    CONTEXT_CASE_IDS,
    FROZEN_CASES,
    KERNEL_SOURCE,
    TARGET_CASE_IDS,
    frozen_cases,
    get_case,
)


# --- case freeze ------------------------------------------------------------


def test_frozen_case_matrix_is_exactly_six() -> None:
    cases = frozen_cases()
    assert len(cases) == 6
    assert [c.case_id for c in cases] == ["t1", "t2", "t3", "t4", "c1", "c2"]
    assert TARGET_CASE_IDS == ("t1", "t2", "t3", "t4")
    assert CONTEXT_CASE_IDS == ("c1", "c2")


def test_frozen_shapes_match_preregistration() -> None:
    expected = {
        "t1": (1, 32, 32, "target"),
        "t2": (1, 128, 64, "target"),
        "t3": (8, 64, 64, "target"),
        "t4": (16, 128, 64, "target"),
        "c1": (64, 128, 64, "context"),
        "c2": (256, 256, 128, "context"),
    }
    for case in FROZEN_CASES:
        batch, in_f, out_f, role = expected[case.case_id]
        assert (case.batch, case.in_features, case.out_features, case.role) == (
            batch,
            in_f,
            out_f,
            role,
        )
        assert case.output_shape == (batch,)
        # Seeds are CASE_SEED_BASE + fixed offsets 1..6 in freeze order.
        offsets = {"t1": 1, "t2": 2, "t3": 3, "t4": 4, "c1": 5, "c2": 6}
        assert case.seed == 20260717 + offsets[case.case_id]


def test_kernel_source_is_phase_a_chain() -> None:
    assert "F.linear" in KERNEL_SOURCE
    assert ".relu()" in KERNEL_SOURCE
    assert "mean(dim=1" in KERNEL_SOURCE
    assert "TensorF32Cpu2D" in KERNEL_SOURCE


def test_get_case_rejects_unknown() -> None:
    with pytest.raises(KeyError, match="unknown frozen"):
        get_case("t9")


# --- scheduling -------------------------------------------------------------


def test_counterbalanced_schedule_is_balanced_for_nine() -> None:
    schedule = _counterbalanced_schedule(9, random.Random(20260717))
    assert len(schedule) == 9
    balance = _schedule_balance(schedule)
    assert balance["is_counterbalanced"] is True
    assert {balance["native_first"], balance["fallback_first"]} == {4, 5}
    assert all(sorted(order) == ["fallback", "native"] for order in schedule)


def test_counterbalanced_extra_lane_is_not_always_native() -> None:
    outcomes = set()
    for seed in range(64):
        balance = _schedule_balance(_counterbalanced_schedule(9, random.Random(seed)))
        outcomes.add((balance["native_first"], balance["fallback_first"]))
    assert (5, 4) in outcomes and (4, 5) in outcomes


def test_counterbalanced_schedule_is_seed_deterministic() -> None:
    assert _counterbalanced_schedule(9, random.Random(7)) == _counterbalanced_schedule(
        9, random.Random(7)
    )


def test_schedule_balance_rejects_malformed_pairs() -> None:
    for schedule in (
        [["native", "native"]] * 5 + [["fallback", "fallback"]] * 4,
        [["native", "sideways"]] * 5 + [["fallback", "native"]] * 4,
        [["native"]] * 5 + [["fallback", "native"]] * 4,
    ):
        balance = _schedule_balance(schedule)
        assert balance["all_pairs_valid"] is False
        assert balance["is_counterbalanced"] is False


# --- calibration + retry with fakes ----------------------------------------


def test_calibrate_common_iterations_grows_until_both_lanes_clear_floor() -> None:
    calls: list[tuple[str, int]] = []

    def run_lane(mode: str, iterations: int) -> int:
        calls.append((mode, iterations))
        # native needs 4 iters to clear 20ms; fallback is always above.
        if mode == "native":
            return 6_000_000 * iterations
        return 25_000_000 * iterations

    iterations = _calibrate_common_iterations(run_lane, target_ns=FLOOR_NS)
    assert iterations == 4
    assert ("native", 1) in calls and ("fallback", 4) in calls


def test_collect_paired_samples_retries_when_any_sample_under_floor() -> None:
    state = {"iterations_seen": []}

    def run_lane(mode: str, iterations: int) -> int:
        state["iterations_seen"].append(iterations)
        # First attempt (iterations=1): under floor. After double: clear.
        if iterations < 2:
            return 5_000_000
        return 25_000_000

    schedule = [["native", "fallback"], ["fallback", "native"]]
    final_iters, samples = _collect_paired_samples_with_retry(
        run_lane,
        schedule,
        iterations=1,
        floor_ns=FLOOR_NS,
        max_retries=4,
    )
    assert final_iters == 2
    assert len(samples) == 2
    assert all(s["native_ns"] >= FLOOR_NS and s["fallback_ns"] >= FLOOR_NS for s in samples)
    assert 1 in state["iterations_seen"] and 2 in state["iterations_seen"]


def test_collect_paired_samples_fail_closed_after_retries() -> None:
    def run_lane(_mode: str, _iterations: int) -> int:
        return 1_000  # always under floor

    with pytest.raises(RuntimeError, match="could not retain every paired sample"):
        _collect_paired_samples_with_retry(
            run_lane,
            [["native", "fallback"]],
            iterations=1,
            floor_ns=FLOOR_NS,
            max_retries=2,
        )


# --- bootstrap determinism --------------------------------------------------


def test_percentile_linear_interpolation_is_exact() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert _percentile(values, 0.0) == 10.0
    assert _percentile(values, 1.0) == 40.0
    assert _percentile(values, 0.5) == 25.0


def test_paired_log_bootstrap_is_seed_deterministic() -> None:
    native = [9.0, 8.0, 10.0, 7.0, 9.5]
    fallback = [10.0, 10.0, 10.0, 10.0, 10.0]
    first = _paired_log_bootstrap(native, fallback, resamples=500, seed=42)
    second = _paired_log_bootstrap(native, fallback, resamples=500, seed=42)
    assert first == second
    low, high = first["ratio_ci95_native_over_fallback"]
    center = first["ratio_geomean_native_over_fallback"]
    assert low <= center <= high
    # speedup is reciprocal of ratio on the log scale
    assert math.isclose(
        first["speedup_geomean_fallback_over_native"],
        1.0 / center,
        rel_tol=1e-12,
    )


# --- eligibility + GO gate --------------------------------------------------


def _eligible_product(
    *,
    native_over_fallback: float = 0.80,
    repetitions: int = 9,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a synthetic product block that passes primary eligibility."""
    schedule = _counterbalanced_schedule(repetitions, random.Random(seed))
    # Per-call ns encode the ratio; scale iterations so *totals* clear the floor.
    fallback_pc = 12_000_000.0
    native_pc = fallback_pc * native_over_fallback
    min_pc = min(native_pc, fallback_pc)
    iterations = max(1, int(math.ceil(FLOOR_NS / min_pc)))
    native_total = int(native_pc * iterations)
    fallback_total = int(fallback_pc * iterations)
    assert min(native_total, fallback_total) >= FLOOR_NS
    samples = []
    for round_index, order in enumerate(schedule):
        samples.append(
            {
                "round": round_index,
                "schedule": list(order),
                "native_ns": native_total,
                "fallback_ns": fallback_total,
                "iterations": iterations,
            }
        )
    boot = _paired_log_bootstrap(
        [native_pc] * repetitions,
        [fallback_pc] * repetitions,
        resamples=200,
        seed=seed + 1000,
    )
    return {
        "iterations": iterations,
        "schedule": [s["schedule"] for s in samples],
        "schedule_balance": _schedule_balance([s["schedule"] for s in samples]),
        "samples": samples,
        "native_median_ns_per_call": native_pc,
        "fallback_median_ns_per_call": fallback_pc,
        "paired_bootstrap": boot,
        "correctness": {
            "digest_match": True,
            "dtype_ok": True,
            "device_ok": True,
            "shape_ok": True,
            "no_grad_ok": True,
            "non_mutation_ok": True,
        },
        "correctness_ok": True,
    }


def _cell(
    case_id: str,
    role: str,
    *,
    native_over_fallback: float = 0.80,
    route_verified: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    case = get_case(case_id)
    product = _eligible_product(native_over_fallback=native_over_fallback, seed=seed)
    return {
        "case_id": case_id,
        "role": role,
        "batch": case.batch,
        "in_features": case.in_features,
        "out_features": case.out_features,
        "seed": case.seed,
        "product": product,
        "route_verified": route_verified,
        "correctness_ok": True,
    }


def test_valid_cell_is_primary_eligible() -> None:
    cell = _cell("t1", "target")
    ok, reasons = _cell_primary_eligible(cell, repetitions=9)
    assert ok is True
    assert reasons == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda c: c.update(route_verified=False), "route-or-provenance-unverified"),
        (lambda c: c.update(correctness_ok=False), "correctness-failed"),
        (
            lambda c: c["product"]["correctness"].update(digest_match=False),
            "correctness-digest-mismatch",
        ),
        (
            lambda c: c["product"]["samples"].pop(),
            "missing-required-sample",
        ),
        (
            lambda c: c["product"]["samples"].__setitem__(
                0,
                {
                    **c["product"]["samples"][0],
                    "native_ns": 1_000,
                    "fallback_ns": 25_000_000,
                },
            ),
            "sample-below-floor",
        ),
        (
            lambda c: c["product"].update(
                samples=[
                    {
                        **c["product"]["samples"][0],
                        "schedule": ["native", "fallback"],
                    }
                ]
                * 9
            ),
            "schedule-not-counterbalanced",
        ),
    ],
)
def test_primary_eligibility_fails_closed(mutate: Any, expected: str) -> None:
    cell = _cell("t1", "target")
    mutate(cell)
    ok, reasons = _cell_primary_eligible(cell, repetitions=9)
    assert ok is False
    assert expected in reasons


def _passing_matrix() -> list[dict[str, Any]]:
    """Synthetic matrix that should clear the preregistered GO thresholds."""
    cells = []
    # Four target cells clearly faster (ratio ~0.75 → speedup ~1.33).
    for index, case_id in enumerate(TARGET_CASE_IDS):
        cells.append(_cell(case_id, "target", native_over_fallback=0.75, seed=10 + index))
    for index, case_id in enumerate(CONTEXT_CASE_IDS):
        cells.append(_cell(case_id, "context", native_over_fallback=0.90, seed=20 + index))
    return cells


def test_expansion_go_gate_passes_on_favorable_matrix() -> None:
    gate = compute_expansion_go_gate(
        _passing_matrix(),
        repetitions=9,
        bootstrap_resamples=500,
        seed=20260717,
    )
    assert gate["verdict"] == "GO"
    assert gate["reasons"] == []
    assert gate["aggregate_geomean_fallback_over_native"] >= GO_AGGREGATE_GM_SPEEDUP_MIN
    assert gate["aggregate_speedup_ci95"][0] >= GO_AGGREGATE_CI_LOW_MIN
    assert gate["target_cells_ci_upper_below_one"] >= 3


def test_expansion_go_gate_fails_when_aggregate_speedup_too_low() -> None:
    # ratio 0.95 → speedup ~1.05 < 1.20
    cells = [
        _cell(cid, "target", native_over_fallback=0.95, seed=i)
        for i, cid in enumerate(TARGET_CASE_IDS)
    ]
    cells += [
        _cell(cid, "context", native_over_fallback=0.95, seed=10 + i)
        for i, cid in enumerate(CONTEXT_CASE_IDS)
    ]
    gate = compute_expansion_go_gate(cells, repetitions=9, bootstrap_resamples=200, seed=1)
    assert gate["verdict"] == "NO-GO"
    assert any("aggregate-geomean-speedup-below" in r for r in gate["reasons"])


def test_expansion_go_gate_fails_when_median_ratio_exceeds_cap() -> None:
    cells = []
    for i, cid in enumerate(TARGET_CASE_IDS):
        # t4 is slower than fallback: median n/f > 1.10
        ratio = 1.20 if cid == "t4" else 0.70
        cells.append(_cell(cid, "target", native_over_fallback=ratio, seed=i))
    cells += [
        _cell(cid, "context", native_over_fallback=0.80, seed=20 + i)
        for i, cid in enumerate(CONTEXT_CASE_IDS)
    ]
    gate = compute_expansion_go_gate(cells, repetitions=9, bootstrap_resamples=200, seed=2)
    assert gate["verdict"] == "NO-GO"
    assert any("target-median-ratio-exceeds-cap:t4" in r for r in gate["reasons"])
    # Cap constant is the preregistered value.
    assert GO_MAX_CELL_MEDIAN_RATIO == 1.10


def test_expansion_go_gate_ignores_context_lane_rescue() -> None:
    """Even huge context-size speedups cannot rescue a failed target aggregate."""
    cells = [
        _cell(cid, "target", native_over_fallback=0.95, seed=i)
        for i, cid in enumerate(TARGET_CASE_IDS)
    ]
    # Context sizes look great — must not flip the verdict.
    cells += [
        _cell(cid, "context", native_over_fallback=0.10, seed=50 + i)
        for i, cid in enumerate(CONTEXT_CASE_IDS)
    ]
    gate = compute_expansion_go_gate(cells, repetitions=9, bootstrap_resamples=200, seed=3)
    assert gate["verdict"] == "NO-GO"
    assert "never contribute" in gate["note"]


def test_expansion_go_gate_requires_all_primary_cells() -> None:
    cells = _passing_matrix()
    cells = [c for c in cells if c["case_id"] != "c2"]  # drop a primary cell
    gate = compute_expansion_go_gate(cells, repetitions=9, bootstrap_resamples=200, seed=4)
    assert gate["verdict"] == "NO-GO"
    assert any("missing-primary-cell:c2" in r for r in gate["reasons"])


# --- smoke / full output ownership -----------------------------------------


def test_require_raises_without_using_assert() -> None:
    _require(True, "ok")
    with pytest.raises(RuntimeError, match="benchmark preflight failed: boom"):
        _require(False, "boom")


def test_authoritative_paths_are_detected() -> None:
    assert _is_authoritative_path(bench.FULL_RESULT) is True
    assert _is_authoritative_path(bench.FULL_REPORT) is True
    assert _is_authoritative_path(bench.EVIDENCE_DIR / "check.json") is True
    assert (
        _is_authoritative_path(Path(tempfile_path := "/tmp/rextio-torch-smoke-x/smoke.json"))
        is False
    )
    del tempfile_path


def test_smoke_output_refuses_to_touch_authoritative_result() -> None:
    result = {"provenance": {"report_evidence_files": {"check.json": "{}"}}}
    with pytest.raises(RuntimeError):
        bench._write_smoke_output(result, bench.FULL_RESULT)
    with pytest.raises(RuntimeError):
        bench._write_smoke_output(result, bench.EVIDENCE_DIR / "check.json")


def test_smoke_writes_to_temp_and_leaves_sentinel_untouched(tmp_path: Path) -> None:
    sentinel = "SENTINEL-FULL-RESULT"
    existed = bench.FULL_RESULT.exists()
    original = bench.FULL_RESULT.read_text() if existed else None
    bench.FULL_RESULT.parent.mkdir(parents=True, exist_ok=True)
    bench.FULL_RESULT.write_text(sentinel, encoding="utf-8")
    try:
        result = {
            "provenance": {"report_evidence_files": {"check.json": "{}"}},
            "ok": True,
        }
        out = bench._write_smoke_output(result, tmp_path / "smoke.json")
        assert out == (tmp_path / "smoke.json").resolve()
        assert json.loads(out.read_text())["ok"] is True
        assert bench.FULL_RESULT.read_text() == sentinel
        assert "report_evidence_files" not in result["provenance"]
    finally:
        if original is not None:
            bench.FULL_RESULT.write_text(original, encoding="utf-8")
        elif bench.FULL_RESULT.exists():
            bench.FULL_RESULT.unlink()


def test_smoke_default_output_is_a_temp_location() -> None:
    result = {"provenance": {}, "ok": True}
    out = bench._write_smoke_output(result, None)
    try:
        assert out.exists()
        assert not out.is_relative_to(bench.RESULTS_DIR.resolve())
    finally:
        shutil.rmtree(out.parent, ignore_errors=True)


def test_full_output_writes_atomic_latest_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    monkeypatch.setattr(bench, "RESULTS_DIR", results)
    monkeypatch.setattr(bench, "FULL_RESULT", results / "latest.json")
    monkeypatch.setattr(bench, "FULL_REPORT", results / "report.md")
    monkeypatch.setattr(bench, "EVIDENCE_DIR", results / "evidence")
    monkeypatch.setattr(
        bench,
        "_HARNESS_OWNED_RESULTS",
        (results / "latest.json", results / "report.md", results / "evidence"),
    )

    cells = _passing_matrix()
    for cell in cells:
        ok, reasons = _cell_primary_eligible(cell, repetitions=9)
        cell["primary_eligible"] = ok
        cell["primary_ineligible_reasons"] = reasons
    gate = compute_expansion_go_gate(cells, repetitions=9, bootstrap_resamples=200, seed=5)
    result = {
        "schema": "rextio-torch-phase-a-product-benchmark-v1",
        "cells": cells,
        "expansion_go_gate": gate,
        "non_claims": bench._non_claims(),
        "provenance": {
            "report_evidence_files": {
                "check.json": '{"ok": true}',
                "build.json": '{"status": "built"}',
            }
        },
    }
    out = bench._write_full_output(result)
    assert out == results / "latest.json"
    assert out.exists()
    assert (results / "report.md").exists()
    assert (results / "evidence" / "check.json").exists()
    assert (results / "evidence" / "build.json").exists()
    loaded = json.loads(out.read_text())
    assert "report_evidence_files" not in loaded["provenance"]
    assert loaded["expansion_go_gate"]["verdict"] in {"GO", "NO-GO"}


# --- fail-closed provenance validators -------------------------------------


def test_core_provenance_accepts_index_install_without_direct_url() -> None:
    assert _validate_core_provenance(None, Path("/tmp/rextio/__init__.py"), None) == "index"


def test_core_provenance_rejects_credentialed_url() -> None:
    with pytest.raises(RuntimeError):
        _validate_core_provenance(
            {
                "url": "https://ghp_secret@github.com/rextio/rextio.git",
                "vcs_info": {"vcs": "git", "commit_id": "abc123"},
            },
            Path("/tmp/rextio/__init__.py"),
            None,
        )


def test_core_version_range_gate() -> None:
    assert bench._rextio_version_supported("0.1.3")
    assert bench._rextio_version_supported("0.1.9")
    assert not bench._rextio_version_supported("0.1.2")
    assert not bench._rextio_version_supported("0.2.0")


def test_plugin_direct_url_validator_accepts_wheel_mode() -> None:
    mode = _validate_plugin_direct_url(
        {
            "url": "https://example/rextio_torch-0.1.0-py3-none-any.whl",
            "archive_info": {"hashes": {"sha256": "abc"}},
        },
        Path("/site-packages/rextio_torch/__init__.py"),
    )
    assert mode == "wheel"


def test_native_artifact_binding_rejects_a_different_file(tmp_path: Path) -> None:
    installed = tmp_path / "generated" / "_rextio_native.so"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"the-real-artifact-bytes")
    imported = tmp_path / "build" / "_rextio_native.so"
    imported.parent.mkdir(parents=True)
    imported.write_bytes(b"the-real-artifact-bytes")
    assert bench._require_bound_native_artifact(str(installed), str(imported)) == imported.resolve()
    stale = tmp_path / "build" / "_rextio_native_stale.so"
    stale.write_bytes(b"a-different-stale-artifact")
    with pytest.raises(RuntimeError):
        bench._require_bound_native_artifact(str(installed), str(stale))
