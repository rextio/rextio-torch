"""Focused Phase B harness contracts; no builds, workers, or timing runs."""

from __future__ import annotations

import importlib
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmarks import bench_phase_b
from benchmarks.bench_phase_b import (
    EVIDENCE_DIR,
    FULL_REPORT,
    FULL_RESULT,
    RESULTS_DIR,
    _is_authoritative_path,
    compute_phase_b_go_gate,
)
from benchmarks.phase_b_cases import (
    CASE_SEED_BASE,
    CONTEXT_CASE_IDS,
    FROZEN_CASES,
    KERNEL_SOURCE,
    TARGET_CASE_IDS,
    frozen_cases,
    get_case,
)


def test_frozen_phase_b_matrix_and_kernel_are_exact() -> None:
    expected = {
        "t1": ("target", 1, 16, 4, 0, 101),
        "t2": ("target", 1, 32, 8, 1, 102),
        "t3": ("target", 4, 32, 16, 0, 103),
        "t4": ("target", 8, 64, 16, 1, 104),
        "c1": ("context", 32, 128, 8, 0, 105),
        "c2": ("context", 64, 128, 16, 1, 106),
    }

    assert TARGET_CASE_IDS == ("t1", "t2", "t3", "t4")
    assert CONTEXT_CASE_IDS == ("c1", "c2")
    assert [case.case_id for case in FROZEN_CASES] == [*TARGET_CASE_IDS, *CONTEXT_CASE_IDS]
    assert frozen_cases() == list(FROZEN_CASES)

    for case in FROZEN_CASES:
        role, batch, width, depth, phase, seed_offset = expected[case.case_id]
        assert (case.role, case.batch, case.width, case.depth, case.phase) == (
            role,
            batch,
            width,
            depth,
            phase,
        )
        assert case.seed == CASE_SEED_BASE + seed_offset
        assert case.x_shape == (batch, width)
        assert case.weight_shape == (width, width)
        assert case.bias_shape == (width,)
        assert case.output_shape == (batch,)
        assert get_case(case.case_id) is case

    assert "for layer in range(depth):" in KERNEL_SOURCE
    assert "if (layer + phase) % 2 == 0:" in KERNEL_SOURCE
    assert KERNEL_SOURCE.count("F.linear(") == 2
    assert KERNEL_SOURCE.count(".relu()") == 2
    assert KERNEL_SOURCE.count("mean(dim=1, keepdim=False)") == 1


def test_phase_b_output_ownership_is_separate_and_fail_closed(tmp_path: Path) -> None:
    assert RESULTS_DIR.name == "results_phase_b"
    assert FULL_RESULT == RESULTS_DIR / "latest.json"
    assert FULL_REPORT == RESULTS_DIR / "report.md"
    assert EVIDENCE_DIR == RESULTS_DIR / "evidence"

    assert _is_authoritative_path(FULL_RESULT)
    assert _is_authoritative_path(FULL_REPORT)
    assert _is_authoritative_path(EVIDENCE_DIR / "check.json")
    assert not _is_authoritative_path(RESULTS_DIR / "smoke.json")
    assert not _is_authoritative_path(RESULTS_DIR.parent / "results" / "latest.json")
    assert not _is_authoritative_path(tmp_path / "latest.json")


def _schedule(comparator: str) -> list[list[str]]:
    return [
        ["native", comparator] if index % 2 == 0 else [comparator, "native"]
        for index in range(9)
    ]


def _comparison(comparator: str, native_over_comparator: float) -> dict[str, Any]:
    comparator_per_call = 10_000_000.0
    native_per_call = comparator_per_call * native_over_comparator
    iterations = 3
    schedule = _schedule(comparator)
    samples = [
        {
            "round": index,
            "schedule": order,
            "iterations": iterations,
            "native_ns": int(native_per_call * iterations),
            "comparator_ns": int(comparator_per_call * iterations),
            f"{comparator}_ns": int(comparator_per_call * iterations),
        }
        for index, order in enumerate(schedule)
    ]
    speedup = 1.0 / native_over_comparator
    bootstrap = {
        "ratio_geomean_native_over_comparator": native_over_comparator,
        "ratio_ci95_native_over_comparator": [
            native_over_comparator,
            native_over_comparator,
        ],
        "speedup_geomean_comparator_over_native": speedup,
        "speedup_ci95_comparator_over_native": [speedup, speedup],
        f"ratio_geomean_native_over_{comparator}": native_over_comparator,
        f"ratio_ci95_native_over_{comparator}": [
            native_over_comparator,
            native_over_comparator,
        ],
        f"speedup_geomean_{comparator}_over_native": speedup,
        f"speedup_ci95_{comparator}_over_native": [speedup, speedup],
    }
    return {
        "iterations": iterations,
        "schedule": schedule,
        "schedule_balance": {
            "native_first": 5,
            "comparator_first": 4,
            f"{comparator}_first": 4,
            "all_pairs_valid": True,
            "is_counterbalanced": True,
        },
        "samples": samples,
        "native_median_ns_per_call": native_per_call,
        "comparator_median_ns_per_call": comparator_per_call,
        f"{comparator}_median_ns_per_call": comparator_per_call,
        "paired_bootstrap": bootstrap,
        "correctness_ok": True,
    }


def _cell(case_id: str, native_over_comparator: float) -> dict[str, Any]:
    case = get_case(case_id)
    return {
        **case.to_dict(),
        "route_verified": True,
        "correctness_ok": True,
        "comparisons_eligible": True,
        "comparisons": {
            "fallback": _comparison("fallback", native_over_comparator),
            "direct_eager": _comparison("direct_eager", native_over_comparator),
        },
        "contexts": {
            "torch_compile_fullgraph": {
                "available": False,
                "correctness_ok": False,
                "warm_median_ns_per_call": None,
                "reason": "synthetic context unavailable",
            }
        },
    }


def test_phase_b_gate_passes_strong_synthetic_results() -> None:
    cells = [_cell(case.case_id, 0.75) for case in FROZEN_CASES]

    gate = compute_phase_b_go_gate(
        cells,
        repetitions=9,
        bootstrap_resamples=1_000,
        seed=CASE_SEED_BASE,
    )

    assert gate["verdict"] == "GO"
    assert gate["reasons"] == []
    for comparator in ("fallback", "direct_eager"):
        result = gate["comparators"][comparator]
        assert result["aggregate_geomean_comparator_over_native"] >= 1.20
        assert result["aggregate_speedup_ci95"][0] >= 1.10
        assert result["target_cells_ci_upper_below_one"] >= 3


def test_phase_b_gate_fails_weak_target_results_even_with_fast_context() -> None:
    cells = [
        _cell(case.case_id, 0.95 if case.role == "target" else 0.20)
        for case in FROZEN_CASES
    ]

    gate = compute_phase_b_go_gate(
        cells,
        repetitions=9,
        bootstrap_resamples=1_000,
        seed=CASE_SEED_BASE,
    )

    assert gate["verdict"] == "NO-GO"
    assert gate["reasons"]
    assert any("aggregate" in reason for reason in gate["reasons"])


def test_import_does_not_execute_timing_or_subprocesses(monkeypatch: Any) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("benchmark execution occurred during import")

    with monkeypatch.context() as patch:
        patch.setattr(time, "perf_counter_ns", fail)
        patch.setattr(subprocess, "run", fail)
        importlib.reload(bench_phase_b)

    # Restore module-local imports after the patched reload for other tests.
    importlib.reload(bench_phase_b)
    assert bench_phase_b.RESULTS_DIR == RESULTS_DIR
