"""Preregistered Phase A product-route benchmark harness.

Primary comparison: the same generated Rextio wrapper in two persistent
processes, differing only by ``REXTIO_NATIVE_MODE`` (native vs fallback).
Build once; import torch before the native extension; compile and first-call
warm-up stay outside steady-state samples.

Direct eager PyTorch and ``torch.compile(fullgraph=True)`` are context-only
lanes. They never enter the Rextio expansion GO gate; unavailable context is
reported honestly.

Do not run a full performance measurement until the protocol document is
accepted and the environment matches the frozen pins. Smoke writes only to
ignored/temp paths and never overwrites authoritative results.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

# Script and ``python -m`` both need the repo root for package imports / workers.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.cases import (  # noqa: E402
    CONTEXT_CASE_IDS,
    FROZEN_CASES,
    KERNEL_SOURCE,
    TARGET_CASE_IDS,
    BenchmarkCase,
    frozen_cases,
    get_case,
)

ROOT = _ROOT
PLUGIN_SRC = ROOT / "src"
CORE_ROOT_ENV = "REXTIO_CORE_ROOT"
REQUIRED_REXTIO_SPEC = ">=0.1.3,<0.2"
REQUIRED_PLUGIN_API = "1.3"
REQUIRED_TORCH = "2.11.0"
REQUIRED_PYTHON = (3, 11)
REQUIRED_TCH = "=0.24.0"
REQUIRED_TCH_FEATURE = "python-extension"
FIXTURE_PACKAGE = "torch_bench_app"
FIXTURE_MODULE = f"{FIXTURE_PACKAGE}.kernels"
FUNCTION_QUALNAME = f"{FIXTURE_MODULE}.inference"
EXPECTED_ROUTE = "native-plugin:rextio-torch"
EXPECTED_CLAIM_RULES = frozenset(
    {
        "rextio-torch/functional-linear-f32-cpu-2d",
        "rextio-torch/tensor-relu-f32-cpu-2d",
        "rextio-torch/tensor-mean-dim1-f32-cpu-2d",
    }
)

RESULTS_DIR = ROOT / "benchmarks" / "results"
FULL_RESULT = RESULTS_DIR / "latest.json"
FULL_REPORT = RESULTS_DIR / "report.md"
EVIDENCE_DIR = RESULTS_DIR / "evidence"
_HARNESS_OWNED_RESULTS = (FULL_RESULT, FULL_REPORT, EVIDENCE_DIR)

# Protocol constants (preregistered 2026-07-17).
DEFAULT_REPETITIONS = 9
DEFAULT_TARGET_MS = 20.0
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SEED = 20260717
FLOOR_NS = 20_000_000  # both lanes must retain samples ≥ 20 ms
CALIBRATION_MAX_RETRIES = 4
MAX_COMMON_ITERATIONS = 1_048_576

# Expansion GO gate (binary; context lanes cannot rescue a failure).
GO_AGGREGATE_GM_SPEEDUP_MIN = 1.20  # geomean(fallback/native) point estimate
GO_AGGREGATE_CI_LOW_MIN = 1.10  # paired-bootstrap 95% lower bound on speedup
GO_MIN_CELLS_CI_UPPER_BELOW_ONE = 3  # of 4 target cells: native/fallback CI high < 1.0
GO_MAX_CELL_MEDIAN_RATIO = 1.10  # no target median(native/fallback) above this

_VALID_SCHEDULE_PAIRS = (("native", "fallback"), ("fallback", "native"))


# ---------------------------------------------------------------------------
# Fail-closed utilities
# ---------------------------------------------------------------------------


def _require(condition: object, message: str) -> None:
    """Fail-closed runtime gate that survives ``python -O`` (no ``assert``)."""
    if not condition:
        raise RuntimeError(f"benchmark preflight failed: {message}")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _normalize_dist(name: str) -> str:
    return name.lower().replace("_", "-")


def _parse_version(value: str) -> tuple[int, int, int]:
    core = value.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return major, minor, patch


def _rextio_version_supported(value: str) -> bool:
    version = _parse_version(value)
    return (0, 1, 3) <= version < (0, 2, 0)


def _optional_core_root() -> Path | None:
    raw = os.environ.get(CORE_ROOT_ENV, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _distribution_version(distribution: str) -> str:
    return importlib_metadata.version(distribution)


def _distribution_direct_url(distribution: str) -> dict | None:
    target = _normalize_dist(distribution)
    for dist in importlib_metadata.distributions():
        name = dist.metadata["Name"]
        if name is None or _normalize_dist(name) != target:
            continue
        raw = dist.read_text("direct_url.json")
        if raw:
            return json.loads(raw)
    return None


def _file_url_to_path(url: str) -> Path:
    parts = urlsplit(url)
    _require(parts.scheme == "file", f"expected a file:// URL, got {url!r}")
    return Path(unquote(parts.path)).resolve()


def _git_sha(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _git_dirty(path: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return bool(status.strip())


def _command_text(command: Sequence[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return completed.stdout.strip() or completed.stderr.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_bound_native_artifact(installed_path: str, imported_file: str) -> Path:
    """Bind the timed native module to this build's exact artifact by bytes."""
    installed = Path(installed_path).resolve()
    imported = Path(imported_file).resolve()
    _require(installed.exists(), f"reported native artifact missing: {installed}")
    _require(imported.exists(), f"imported native module missing: {imported}")
    _require(
        _sha256_file(imported) == _sha256_file(installed),
        f"timed native module {imported} is not byte-identical to the build artifact {installed}",
    )
    return imported


# ---------------------------------------------------------------------------
# Schedule / statistics (pure; unit-tested with fakes)
# ---------------------------------------------------------------------------


def _counterbalanced_schedule(repetitions: int, rng: random.Random) -> list[list[str]]:
    """Return a genuinely counterbalanced native/fallback order schedule.

    Native-first and fallback-first counts differ by at most one. For odd
    counts the extra first position is assigned from the per-cell seed (not
    always native). Orders are then seed-shuffled.
    """
    base = repetitions // 2
    native_first = base
    fallback_first = base
    if repetitions % 2 == 1:
        if rng.random() < 0.5:
            native_first += 1
        else:
            fallback_first += 1
    orders = [["native", "fallback"] for _ in range(native_first)]
    orders += [["fallback", "native"] for _ in range(fallback_first)]
    rng.shuffle(orders)
    return orders


def _schedule_balance(schedule: Sequence[Sequence[str]]) -> dict[str, Any]:
    all_pairs_valid = all(
        isinstance(order, (list, tuple)) and tuple(order) in _VALID_SCHEDULE_PAIRS
        for order in schedule
    )
    native_first = sum(1 for order in schedule if len(order) >= 1 and order[0] == "native")
    fallback_first = sum(1 for order in schedule if len(order) >= 1 and order[0] == "fallback")
    return {
        "native_first": native_first,
        "fallback_first": fallback_first,
        "all_pairs_valid": all_pairs_valid,
        "is_counterbalanced": (
            all_pairs_valid
            and len(schedule) > 0
            and native_first + fallback_first == len(schedule)
            and abs(native_first - fallback_first) <= 1
        ),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_log_bootstrap(
    native: Sequence[float],
    fallback: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired bootstrap over log(native/fallback); also report fallback/native.

    Returns point estimates from the observed pairs and a 95% CI from
    *resamples* with-replacement draws (deterministic for a fixed seed).
    """
    if len(native) != len(fallback) or not native:
        raise ValueError("paired bootstrap requires equal non-empty sample lists")
    ratios = [float(n) / float(f) for n, f in zip(native, fallback, strict=True)]
    logs = [math.log(r) for r in ratios]
    speedups = [1.0 / r for r in ratios]
    rng = random.Random(seed)
    boot_logs: list[float] = []
    for _ in range(resamples):
        draw = [logs[rng.randrange(len(logs))] for _ in logs]
        boot_logs.append(statistics.fmean(draw))
    boot_logs.sort()
    low = boot_logs[int(0.025 * resamples)]
    high = boot_logs[min(resamples - 1, int(0.975 * resamples))]
    center = statistics.fmean(logs)
    return {
        "method": (f"paired bootstrap mean log(native/fallback), {resamples} resamples"),
        "seed": seed,
        "resamples": resamples,
        "log_ratio_mean": center,
        "log_ratio_ci95": [low, high],
        "ratio_geomean_native_over_fallback": math.exp(center),
        "ratio_ci95_native_over_fallback": [math.exp(low), math.exp(high)],
        "speedup_geomean_fallback_over_native": math.exp(-center),
        "speedup_ci95_fallback_over_native": [math.exp(-high), math.exp(-low)],
        "median_native_over_fallback": statistics.median(ratios),
        "median_fallback_over_native": statistics.median(speedups),
    }


def _calibrate_iterations(
    run: Callable[[int], int],
    *,
    target_ns: int,
    max_iterations: int = MAX_COMMON_ITERATIONS,
) -> int:
    """Grow an iteration count until a single timed batch meets *target_ns*."""
    iterations = 1
    while True:
        elapsed = run(iterations)
        if elapsed >= target_ns:
            return iterations
        if iterations >= max_iterations:
            raise RuntimeError(
                f"could not calibrate iterations to {target_ns} ns "
                f"(elapsed={elapsed}, iterations={iterations})"
            )
        iterations *= 2


def _calibrate_common_iterations(
    run_lane: Callable[[str, int], int],
    *,
    target_ns: int,
    max_iterations: int = MAX_COMMON_ITERATIONS,
) -> int:
    """Grow a shared iteration count until both lanes meet *target_ns*."""
    iterations = 1
    while True:
        native_ns = run_lane("native", iterations)
        fallback_ns = run_lane("fallback", iterations)
        if min(native_ns, fallback_ns) >= target_ns:
            return iterations
        if iterations >= max_iterations:
            raise RuntimeError(
                f"could not calibrate common iterations to {target_ns} ns "
                f"(native={native_ns}, fallback={fallback_ns}, iterations={iterations})"
            )
        iterations *= 2


def _collect_paired_samples_with_retry(
    run_lane: Callable[[str, int], int],
    schedule: Sequence[Sequence[str]],
    iterations: int,
    *,
    floor_ns: int,
    max_retries: int = CALIBRATION_MAX_RETRIES,
) -> tuple[int, list[dict[str, Any]]]:
    """Collect one sample per schedule entry; retry with 2× iterations if any lane is under floor.

    Both lanes share the same common iteration count. A fast sample below
    *floor_ns* invalidates the attempt; every pair is recollected after doubling.
    """
    current = iterations
    for _attempt in range(max_retries):
        samples: list[dict[str, Any]] = []
        all_above = True
        for round_index, order in enumerate(schedule):
            outcomes: dict[str, int] = {}
            for lane in order:
                outcomes[lane] = run_lane(str(lane), current)
            native_ns = outcomes["native"]
            fallback_ns = outcomes["fallback"]
            if min(native_ns, fallback_ns) < floor_ns:
                all_above = False
            samples.append(
                {
                    "round": round_index,
                    "schedule": list(order),
                    "native_ns": native_ns,
                    "fallback_ns": fallback_ns,
                    "iterations": current,
                }
            )
        if all_above:
            return current, samples
        current *= 2
    raise RuntimeError(
        f"could not retain every paired sample above {floor_ns} ns after {max_retries} retries"
    )


# ---------------------------------------------------------------------------
# Expansion GO gate (pure)
# ---------------------------------------------------------------------------


def _cell_primary_eligible(cell: Mapping[str, Any], *, repetitions: int) -> tuple[bool, list[str]]:
    """Fail-closed provenance/correctness eligibility for one primary cell."""
    reasons: list[str] = []
    product = cell.get("product") or {}

    if not cell.get("route_verified", False):
        reasons.append("route-or-provenance-unverified")
    if not cell.get("correctness_ok", False):
        reasons.append("correctness-failed")

    checks = product.get("correctness") or {}
    if checks.get("digest_match") is not True:
        reasons.append("correctness-digest-mismatch")
    for key in (
        "dtype_ok",
        "device_ok",
        "shape_ok",
        "no_grad_ok",
        "non_mutation_ok",
    ):
        if checks.get(key) is not True:
            reasons.append(f"correctness-{key.replace('_', '-')}-failed")

    samples = product.get("samples") or []
    if len(samples) != repetitions:
        reasons.append("missing-required-sample")
    for sample in samples:
        if not (
            _finite(sample.get("native_ns"))
            and _finite(sample.get("fallback_ns"))
            and float(sample["native_ns"]) > 0
            and float(sample["fallback_ns"]) > 0
        ):
            reasons.append("nonpositive-or-nonfinite-sample")
            break
        if min(float(sample["native_ns"]), float(sample["fallback_ns"])) < FLOOR_NS:
            reasons.append("sample-below-floor")
            break

    schedule = [sample.get("schedule", []) for sample in samples]
    balance = _schedule_balance(schedule)
    if len(schedule) != repetitions or not balance["is_counterbalanced"]:
        reasons.append("schedule-not-counterbalanced")

    for lane in ("native", "fallback"):
        median_key = f"{lane}_median_ns_per_call"
        if not _finite(product.get(median_key)):
            reasons.append("nonfinite-median")
            break

    boot = product.get("paired_bootstrap") or {}
    ratio = boot.get("ratio_geomean_native_over_fallback")
    low = (boot.get("ratio_ci95_native_over_fallback") or [None, None])[0]
    high = (boot.get("ratio_ci95_native_over_fallback") or [None, None])[1]
    if not (_finite(ratio) and _finite(low) and _finite(high)) or float(ratio) <= 0:
        reasons.append("missing-or-nonfinite-bootstrap")
    elif float(low) > float(high):
        reasons.append("reversed-ci-bounds")

    return (not reasons, reasons)


def compute_expansion_go_gate(
    cells: Sequence[Mapping[str, Any]],
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Binary expansion GO gate over the four target cells.

    Context sizes and context lanes never contribute. Any failure is NO-GO.
    """
    by_id = {str(cell["case_id"]): cell for cell in cells}
    reasons: list[str] = []

    # Every primary cell (all six) must be provenance/correctness eligible.
    primary_eligible: dict[str, bool] = {}
    primary_reasons: dict[str, list[str]] = {}
    for case in FROZEN_CASES:
        cell = by_id.get(case.case_id)
        if cell is None:
            primary_eligible[case.case_id] = False
            primary_reasons[case.case_id] = ["missing-cell"]
            reasons.append(f"missing-primary-cell:{case.case_id}")
            continue
        ok, cell_reasons = _cell_primary_eligible(cell, repetitions=repetitions)
        primary_eligible[case.case_id] = ok
        primary_reasons[case.case_id] = cell_reasons
        if not ok:
            reasons.append(f"primary-ineligible:{case.case_id}:{','.join(cell_reasons)}")

    target_cells = [by_id[cid] for cid in TARGET_CASE_IDS if cid in by_id]
    if len(target_cells) != len(TARGET_CASE_IDS):
        reasons.append("missing-target-cells")

    per_cell: dict[str, Any] = {}
    cell_speedups: list[float] = []
    cells_ci_upper_below_one = 0
    for cell in target_cells:
        case_id = str(cell["case_id"])
        product = cell.get("product") or {}
        boot = product.get("paired_bootstrap") or {}
        ratio_med = boot.get("median_native_over_fallback")
        if not _finite(ratio_med):
            # Fall back to sample medians when bootstrap block is incomplete.
            samples = product.get("samples") or []
            if samples:
                native_pc = [
                    float(s["native_ns"]) / max(int(s.get("iterations", 1)), 1) for s in samples
                ]
                fallback_pc = [
                    float(s["fallback_ns"]) / max(int(s.get("iterations", 1)), 1) for s in samples
                ]
                ratio_med = statistics.median(
                    n / f for n, f in zip(native_pc, fallback_pc, strict=True)
                )
            else:
                ratio_med = float("nan")
        ci = boot.get("ratio_ci95_native_over_fallback") or [None, None]
        speedup_gm = boot.get("speedup_geomean_fallback_over_native")
        if _finite(speedup_gm):
            cell_speedups.append(float(speedup_gm))
        ci_high = ci[1]
        below = _finite(ci_high) and float(ci_high) < 1.0
        if below:
            cells_ci_upper_below_one += 1
        if _finite(ratio_med) and float(ratio_med) > GO_MAX_CELL_MEDIAN_RATIO:
            reasons.append(f"target-median-ratio-exceeds-cap:{case_id}:{float(ratio_med):.6f}")
        per_cell[case_id] = {
            "median_native_over_fallback": ratio_med,
            "ratio_ci95_native_over_fallback": list(ci),
            "speedup_geomean_fallback_over_native": speedup_gm,
            "ci_upper_below_one": below,
            "primary_eligible": primary_eligible.get(case_id, False),
        }

    if cells_ci_upper_below_one < GO_MIN_CELLS_CI_UPPER_BELOW_ONE:
        reasons.append(
            f"fewer-than-{GO_MIN_CELLS_CI_UPPER_BELOW_ONE}-target-cells-with-ci-upper-below-one:"
            f"{cells_ci_upper_below_one}"
        )

    aggregate_gm: float | None = None
    aggregate_ci: list[float | None] = [None, None]
    if len(cell_speedups) == len(TARGET_CASE_IDS):
        aggregate_gm = math.exp(statistics.fmean(math.log(s) for s in cell_speedups))
        if aggregate_gm < GO_AGGREGATE_GM_SPEEDUP_MIN:
            reasons.append(
                f"aggregate-geomean-speedup-below-{GO_AGGREGATE_GM_SPEEDUP_MIN}:{aggregate_gm:.6f}"
            )
        # Aggregate paired bootstrap: resample pairs within each target cell,
        # recompute cell geomean speedups, then the four-cell geomean.
        rng = random.Random(seed + 17)
        boot_agg: list[float] = []
        cell_logs: list[list[float]] = []
        for cell in target_cells:
            samples = (cell.get("product") or {}).get("samples") or []
            logs = [math.log(float(s["native_ns"]) / float(s["fallback_ns"])) for s in samples]
            cell_logs.append(logs)
        if all(len(logs) == repetitions for logs in cell_logs):
            for _ in range(bootstrap_resamples):
                speedups = []
                for logs in cell_logs:
                    draw = [logs[rng.randrange(len(logs))] for _ in logs]
                    # mean log(native/fallback) → speedup = exp(-mean)
                    speedups.append(math.exp(-statistics.fmean(draw)))
                boot_agg.append(math.exp(statistics.fmean(math.log(s) for s in speedups)))
            boot_agg.sort()
            low = boot_agg[int(0.025 * bootstrap_resamples)]
            high = boot_agg[min(bootstrap_resamples - 1, int(0.975 * bootstrap_resamples))]
            aggregate_ci = [low, high]
            if low < GO_AGGREGATE_CI_LOW_MIN:
                reasons.append(
                    f"aggregate-bootstrap-ci-low-below-{GO_AGGREGATE_CI_LOW_MIN}:{low:.6f}"
                )
        else:
            reasons.append("aggregate-bootstrap-incomplete-samples")
    else:
        reasons.append("aggregate-speedup-unavailable")

    verdict = "GO" if not reasons else "NO-GO"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "thresholds": {
            "aggregate_geomean_fallback_over_native_min": GO_AGGREGATE_GM_SPEEDUP_MIN,
            "aggregate_bootstrap_ci95_low_min": GO_AGGREGATE_CI_LOW_MIN,
            "min_target_cells_ci_upper_native_over_fallback_below_one": (
                GO_MIN_CELLS_CI_UPPER_BELOW_ONE
            ),
            "max_target_median_native_over_fallback": GO_MAX_CELL_MEDIAN_RATIO,
        },
        "primary_eligible": primary_eligible,
        "primary_ineligible_reasons": primary_reasons,
        "target_cells_ci_upper_below_one": cells_ci_upper_below_one,
        "per_cell": per_cell,
        "aggregate_geomean_fallback_over_native": aggregate_gm,
        "aggregate_speedup_ci95": aggregate_ci,
        "note": (
            "Context sizes (c1, c2) and context lanes (eager, torch.compile) "
            "never contribute to this gate."
        ),
    }


# ---------------------------------------------------------------------------
# Provenance / build environment
# ---------------------------------------------------------------------------


def _configure_libtorch_env() -> dict[str, str]:
    """Point tch/PyO3 at this project's CPython 3.11 + torch 2.11.0 venv.

    ``torch-sys`` 0.24 with ``LIBTORCH_USE_PYTORCH=1`` shells out to bare
    ``python`` (when ``VIRTUAL_ENV`` is set) or ``python3`` otherwise. Put the
    dedicated venv's ``bin`` first on ``PATH`` and set ``VIRTUAL_ENV``.
    ``LIBTORCH_BYPASS_VERSION_CHECK`` is never accepted.
    """
    venv_dir = Path(sys.prefix).resolve()
    venv_bin = venv_dir / "bin"
    python = Path(sys.executable).resolve()
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["PYO3_PYTHON"] = str(python)
    env["LIBTORCH_USE_PYTORCH"] = "1"
    env.pop("LIBTORCH_BYPASS_VERSION_CHECK", None)
    _require(
        "LIBTORCH_BYPASS_VERSION_CHECK" not in env,
        "LIBTORCH_BYPASS_VERSION_CHECK must not be set",
    )
    probe = subprocess.run(
        ["python", "-c", "import torch; print(torch.__version__.split('+')[0])"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    _require(probe.returncode == 0, f"torch probe failed: {probe.stderr}")
    version = probe.stdout.strip()
    _require(
        version == REQUIRED_TORCH,
        f"torch-sys build python must be torch {REQUIRED_TORCH}; PATH python reports {version!r}",
    )
    return env


def _validate_core_provenance(
    direct_url: dict | None,
    core_file: Path,
    core_root: Path | None,
) -> str:
    if direct_url is None:
        return "index"
    if direct_url.get("dir_info", {}).get("editable"):
        checkout = _file_url_to_path(direct_url["url"])
        _require(
            core_file.is_relative_to(checkout),
            f"imported rextio {core_file} is not under editable checkout {checkout}",
        )
        if core_root is not None:
            _require(
                checkout.resolve() == core_root.resolve(),
                f"editable core checkout {checkout} != {CORE_ROOT_ENV}={core_root}",
            )
        return "editable"
    if "vcs_info" in direct_url:
        _require("@" not in urlsplit(direct_url["url"]).netloc, "core URL leaks a credential")
        _require(
            "ghp_" not in direct_url["url"] and "x-access-token" not in direct_url["url"],
            "core URL leaks a credential",
        )
        _require(direct_url["vcs_info"].get("vcs") == "git", "core direct URL is not git")
        _require(bool(direct_url["vcs_info"].get("commit_id")), "core VCS commit is missing")
        return "vcs"
    if "archive_info" in direct_url:
        wheel = direct_url["url"].rsplit("/", 1)[-1]
        _require(
            "rextio-" in wheel or wheel.endswith(".whl") or wheel.endswith(".tar.gz"),
            f"core archive does not look like a rextio distribution: {wheel}",
        )
        return "wheel"
    raise RuntimeError(f"benchmark preflight failed: unrecognized core provenance: {direct_url}")


def _validate_plugin_direct_url(direct_url: dict, plugin_file: Path) -> str:
    if direct_url.get("dir_info", {}).get("editable"):
        checkout = _file_url_to_path(direct_url["url"])
        _require(
            checkout.resolve() == ROOT.resolve(),
            f"editable plugin checkout {checkout} != this checkout {ROOT}",
        )
        _require(
            plugin_file.is_relative_to(checkout / "src"),
            f"imported rextio_torch {plugin_file} is not under {checkout / 'src'}",
        )
        return "editable"
    if "archive_info" in direct_url:
        wheel = direct_url["url"].rsplit("/", 1)[-1]
        _require(wheel.endswith(".whl"), f"plugin archive is not a wheel: {wheel}")
        _require("rextio_torch-" in wheel, f"plugin wheel is not rextio_torch: {wheel}")
        _require(bool(direct_url["archive_info"].get("hashes")), "plugin wheel has no hash")
        return "wheel"
    raise RuntimeError(f"benchmark preflight failed: unrecognized plugin provenance: {direct_url}")


def _preflight() -> dict[str, Any]:
    """Reject invalid state before building or timing (survives ``python -O``)."""
    _require(
        sys.version_info[:2] == REQUIRED_PYTHON,
        f"CPython {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} required, found {sys.version}",
    )
    import torch
    import rextio
    import rextio_torch
    from rextio.plugins.api import PLUGIN_API_VERSION
    from rextio_torch.plugin import plugin as this_plugin

    torch_version = torch.__version__.split("+", 1)[0]
    _require(
        torch_version == REQUIRED_TORCH,
        f"torch is {torch.__version__!r}, need {REQUIRED_TORCH}",
    )
    _require(
        PLUGIN_API_VERSION == REQUIRED_PLUGIN_API,
        f"core advertises plugin API {PLUGIN_API_VERSION!r}, need {REQUIRED_PLUGIN_API!r}",
    )
    core_version = _distribution_version("rextio")
    _require(
        _rextio_version_supported(core_version),
        f"rextio {core_version!r} is outside the supported range {REQUIRED_REXTIO_SPEC}",
    )

    # Plugin factory declares the exact tch pin; surface it for provenance.
    crates = list(this_plugin().crate_dependencies())
    tch = next((c for c in crates if c.name == "tch"), None)
    _require(tch is not None, "plugin does not declare a tch rust dependency")
    _require(tch.version == REQUIRED_TCH, f"tch pin is {tch.version!r}, need {REQUIRED_TCH!r}")
    _require(
        REQUIRED_TCH_FEATURE in tch.features,
        f"tch features {tch.features!r} must include {REQUIRED_TCH_FEATURE!r}",
    )

    _require(not _git_dirty(ROOT), "plugin worktree is dirty")
    plugin_sha = _git_sha(ROOT)

    core_root = _optional_core_root()
    core_sha: str | None = None
    if core_root is not None:
        _require(core_root.is_dir(), f"{CORE_ROOT_ENV}={core_root} is not a directory")
        _require(not _git_dirty(core_root), f"core worktree at {core_root} is dirty")
        core_sha = _git_sha(core_root)

    core_file = Path(rextio.__file__).resolve()
    plugin_file = Path(rextio_torch.__file__).resolve()
    core_direct_url = _distribution_direct_url("rextio")
    core_mode = _validate_core_provenance(core_direct_url, core_file, core_root)
    plugin_direct_url = _distribution_direct_url("rextio-torch")
    _require(plugin_direct_url is not None, "plugin has no direct_url.json provenance")
    plugin_mode = _validate_plugin_direct_url(plugin_direct_url, plugin_file)

    if core_sha is None and core_direct_url and "vcs_info" in core_direct_url:
        core_sha = core_direct_url["vcs_info"].get("commit_id")

    entry_points = [
        {"name": ep.name, "value": ep.value, "dist": ep.dist.name if ep.dist else None}
        for ep in importlib_metadata.entry_points(group="rextio.plugins")
        if ep.name == "rextio-torch"
    ]
    _require(
        entry_points
        and all(
            ep["value"] == "rextio_torch.plugin:plugin" and ep["dist"] == "rextio-torch"
            for ep in entry_points
        ),
        f"rextio-torch entry point is not provided by this checkout: {entry_points}",
    )
    loaded = [
        ep
        for ep in importlib_metadata.entry_points(group="rextio.plugins")
        if ep.name == "rextio-torch"
    ]
    _require(
        all(ep.load() is this_plugin for ep in loaded),
        "rextio-torch entry point does not load this exact plugin object",
    )

    harness_path = Path(__file__).resolve()
    cases_path = harness_path.with_name("cases.py")
    return {
        "core_version": core_version,
        "core_sha": core_sha,
        "core_dirty": False,
        "plugin_sha": plugin_sha,
        "plugin_dirty": False,
        "core_install_mode": core_mode,
        "plugin_install_mode": plugin_mode,
        "core_import_file": str(core_file),
        "plugin_import_file": str(plugin_file),
        "core_direct_url": core_direct_url,
        "plugin_direct_url": plugin_direct_url,
        "plugin_api_version": PLUGIN_API_VERSION,
        "required_rextio_spec": REQUIRED_REXTIO_SPEC,
        "required_torch": REQUIRED_TORCH,
        "required_tch": REQUIRED_TCH,
        "required_tch_feature": REQUIRED_TCH_FEATURE,
        "selected_entry_points": entry_points,
        "harness_sha256": _sha256_file(harness_path),
        "cases_sha256": _sha256_file(cases_path),
        "harness_manifest_sha256": _sha256_files([harness_path, cases_path]),
        "torch": torch.__version__,
        "rextio_torch_version": rextio_torch.__version__,
        "python": sys.version,
        "python_executable": sys.executable,
        "rustc": _command_text(["rustc", "--version"]),
        "cargo": _command_text(["cargo", "--version"]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "libtorch_use_pytorch": "1",
        "libtorch_bypass_version_check": False,
    }


def _write_project(root: Path) -> None:
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-torch"]\n',
        encoding="utf-8",
    )
    package = root / "src" / FIXTURE_PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNEL_SOURCE, encoding="utf-8")


# ---------------------------------------------------------------------------
# Worker process (persistent native or fallback)
# ---------------------------------------------------------------------------


def _import_torch_configured() -> Any:
    """Import torch before any native extension; pin threads and seeds."""
    import torch

    # interop thread count can only be set once per process.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_num_threads(1)
    return torch


def _build_tensors(torch: Any, case: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    torch.manual_seed(int(case["seed"]))
    batch = int(case["batch"])
    in_features = int(case["in_features"])
    out_features = int(case["out_features"])
    x = torch.randn(batch, in_features, dtype=torch.float32)
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    bias = torch.randn(out_features, dtype=torch.float32)
    return x, weight, bias


def _tensor_digest(torch: Any, tensor: Any) -> str:
    payload = hashlib.sha256()
    payload.update(type(tensor).__qualname__.encode())
    payload.update(str(tensor.dtype).encode())
    payload.update(str(tensor.device).encode())
    payload.update(str(tuple(tensor.shape)).encode())
    payload.update(str(bool(tensor.requires_grad)).encode())
    payload.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return payload.hexdigest()


def _correctness_payload(
    torch: Any,
    result: Any,
    *,
    x: Any,
    weight: Any,
    bias: Any,
    x_snap: Any,
    w_snap: Any,
    b_snap: Any,
    expected_batch: int,
) -> dict[str, Any]:
    dtype_ok = result.dtype == torch.float32
    device_ok = result.device.type == "cpu"
    shape_ok = tuple(result.shape) == (expected_batch,)
    no_grad_ok = result.requires_grad is False
    non_mutation_ok = bool(
        torch.equal(x, x_snap) and torch.equal(weight, w_snap) and torch.equal(bias, b_snap)
    )
    return {
        "digest": _tensor_digest(torch, result),
        "dtype_ok": dtype_ok,
        "device_ok": device_ok,
        "shape_ok": shape_ok,
        "no_grad_ok": no_grad_ok,
        "non_mutation_ok": non_mutation_ok,
        "dtype": str(result.dtype),
        "device": str(result.device),
        "shape": list(result.shape),
        "requires_grad": bool(result.requires_grad),
    }


def _worker_main(args: argparse.Namespace) -> int:
    os.environ["REXTIO_NATIVE_MODE"] = args.mode
    # Torch must load before the generated native extension.
    torch = _import_torch_configured()
    sys.path.insert(0, args.build_python)
    module = importlib.import_module(FIXTURE_MODULE)
    function = getattr(module, "inference")

    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "close":
            return 0
        case = request["case"]
        iterations = int(request.get("iterations", 1))
        x, weight, bias = _build_tensors(torch, case)
        x_snap = x.detach().clone()
        w_snap = weight.detach().clone()
        b_snap = bias.detach().clone()

        # Untimed first materialization for correctness (also a per-request warm path).
        verification = function(x, weight, bias)
        correctness = _correctness_payload(
            torch,
            verification,
            x=x,
            weight=weight,
            bias=bias,
            x_snap=x_snap,
            w_snap=w_snap,
            b_snap=b_snap,
            expected_batch=int(case["batch"]),
        )
        del verification

        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        start = time.perf_counter_ns()
        try:
            for _ in range(iterations):
                materialized = function(x, weight, bias)
                # Include result destruction in every timed iteration.
                del materialized
        finally:
            elapsed = time.perf_counter_ns() - start
            if enabled:
                gc.enable()
        response = {
            "elapsed_ns": elapsed,
            "iterations": iterations,
            "mode": args.mode,
            "correctness": correctness,
        }
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


class ModeWorker:
    """One persistent generated-wrapper process fixed to native or fallback."""

    def __init__(self, mode: str, build_python: Path, env: Mapping[str, str]) -> None:
        worker_env = dict(env)
        worker_env["PYTHONHASHSEED"] = "0"
        worker_env["REXTIO_NATIVE_MODE"] = mode
        path_parts = [str(PLUGIN_SRC)]
        core_root = _optional_core_root()
        if core_root is not None:
            path_parts.insert(0, str(core_root / "src"))
        existing = worker_env.get("PYTHONPATH")
        if existing:
            path_parts.append(existing)
        worker_env["PYTHONPATH"] = os.pathsep.join(path_parts)
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--mode",
                mode,
                "--build-python",
                str(build_python),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=worker_env,
        )
        self.mode = mode

    def run(self, case: BenchmarkCase, iterations: int) -> dict[str, Any]:
        """Run one timed request in this persistent mode process."""
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"{self.mode} worker pipes are not available")
        request = {"op": "run", "case": case.to_dict(), "iterations": iterations}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise RuntimeError(f"{self.mode} benchmark worker exited: {stderr}")
        return json.loads(line)

    def close(self) -> None:
        """Shut the worker down without leaving a child process."""
        if self.process.poll() is None and self.process.stdin is not None:
            self.process.stdin.write('{"op":"close"}\n')
            self.process.stdin.flush()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


# ---------------------------------------------------------------------------
# Context-only lanes (eager + torch.compile)
# ---------------------------------------------------------------------------


def _context_lanes(
    case: BenchmarkCase,
    *,
    repetitions: int,
    target_ns: int,
    seed: int,
) -> dict[str, Any]:
    """Measure direct eager and torch.compile as context-only (never GO gate)."""
    torch = _import_torch_configured()
    torch.manual_seed(case.seed)
    x, weight, bias = _build_tensors(torch, case.to_dict())

    def eager() -> Any:
        return torch.nn.functional.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)

    # --- eager ---
    try:
        # Warm outside samples.
        warm = eager()
        del warm
        loops = _calibrate_iterations(
            lambda iterations: _timed_batch_local(eager, iterations),
            target_ns=target_ns,
        )
        samples_ns = [_timed_batch_local(eager, loops) for _ in range(repetitions)]
        per_call = [ns / loops for ns in samples_ns]
        eager_block: dict[str, Any] = {
            "available": True,
            "classification": "context-only; not a Rextio target claim",
            "compile_ns": None,
            "loops": loops,
            "samples_ns_total": samples_ns,
            "samples_ns_per_call": per_call,
            "median_ns_per_call": statistics.median(per_call),
            "correctness_digest": _tensor_digest(torch, eager()),
        }
    except Exception as exc:  # noqa: BLE001 — context must report honestly
        eager_block = {
            "available": False,
            "classification": "context-only; not a Rextio target claim",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    # --- torch.compile(fullgraph=True) ---
    try:
        compiled = torch.compile(eager, fullgraph=True)
        compile_start = time.perf_counter_ns()
        first = compiled()
        compile_ns = time.perf_counter_ns() - compile_start
        del first
        # Re-specialization / second-call edge: one more untimed call.
        respec_start = time.perf_counter_ns()
        second = compiled()
        respec_ns = time.perf_counter_ns() - respec_start
        del second

        def compiled_call() -> Any:
            return compiled()

        loops = _calibrate_iterations(
            lambda iterations: _timed_batch_local(compiled_call, iterations),
            target_ns=target_ns,
        )
        samples_ns = [_timed_batch_local(compiled_call, loops) for _ in range(repetitions)]
        per_call = [ns / loops for ns in samples_ns]
        compile_block: dict[str, Any] = {
            "available": True,
            "classification": "context-only; not a Rextio target claim",
            "fullgraph": True,
            "compile_and_first_call_ns": compile_ns,
            "second_call_ns": respec_ns,
            "warm_loops": loops,
            "warm_samples_ns_total": samples_ns,
            "warm_samples_ns_per_call": per_call,
            "warm_median_ns_per_call": statistics.median(per_call),
            "correctness_digest": _tensor_digest(torch, compiled_call()),
        }
    except Exception as exc:  # noqa: BLE001 — context must report honestly
        compile_block = {
            "available": False,
            "classification": "context-only; not a Rextio target claim",
            "fullgraph": True,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    return {
        "eager": eager_block,
        "torch_compile_fullgraph": compile_block,
        "seed": seed,
        "case_id": case.case_id,
    }


def _timed_batch_local(function: Callable[[], object], loops: int) -> int:
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        result: object | None = None
        for _ in range(loops):
            result = function()
        del result
        return time.perf_counter_ns() - start
    finally:
        if enabled:
            gc.enable()


# ---------------------------------------------------------------------------
# Cell measurement
# ---------------------------------------------------------------------------


def _measure_product_cell(
    case: BenchmarkCase,
    workers: dict[str, ModeWorker],
    *,
    repetitions: int,
    target_ns: int,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    # First-call warm-up outside steady-state (per lane, untimed for samples).
    warmups = {mode: workers[mode].run(case, 1) for mode in ("native", "fallback")}
    native_c = warmups["native"]["correctness"]
    fallback_c = warmups["fallback"]["correctness"]
    digest_match = native_c["digest"] == fallback_c["digest"]
    correctness_ok = digest_match and all(
        native_c[key] and fallback_c[key]
        for key in ("dtype_ok", "device_ok", "shape_ok", "no_grad_ok", "non_mutation_ok")
    )

    def run_lane(mode: str, iterations: int) -> int:
        return int(workers[mode].run(case, iterations)["elapsed_ns"])

    iterations = _calibrate_common_iterations(run_lane, target_ns=target_ns)
    rng = random.Random(seed)
    schedule = _counterbalanced_schedule(repetitions, rng)
    iterations, samples = _collect_paired_samples_with_retry(
        run_lane,
        schedule,
        iterations,
        floor_ns=FLOOR_NS,
    )

    native_per_call = [s["native_ns"] / iterations for s in samples]
    fallback_per_call = [s["fallback_ns"] / iterations for s in samples]
    boot = _paired_log_bootstrap(
        native_per_call,
        fallback_per_call,
        resamples=bootstrap_resamples,
        seed=seed + 1000,
    )
    # Re-verify digests from warmups already recorded; attach to product.
    correctness = {
        "native": native_c,
        "fallback": fallback_c,
        "digest_match": digest_match,
        "dtype_ok": bool(native_c["dtype_ok"] and fallback_c["dtype_ok"]),
        "device_ok": bool(native_c["device_ok"] and fallback_c["device_ok"]),
        "shape_ok": bool(native_c["shape_ok"] and fallback_c["shape_ok"]),
        "no_grad_ok": bool(native_c["no_grad_ok"] and fallback_c["no_grad_ok"]),
        "non_mutation_ok": bool(native_c["non_mutation_ok"] and fallback_c["non_mutation_ok"]),
    }
    return {
        "iterations": iterations,
        "schedule": [s["schedule"] for s in samples],
        "schedule_balance": _schedule_balance([s["schedule"] for s in samples]),
        "samples": samples,
        "samples_ns_per_call": {
            "native": native_per_call,
            "fallback": fallback_per_call,
        },
        "native_median_ns_per_call": statistics.median(native_per_call),
        "fallback_median_ns_per_call": statistics.median(fallback_per_call),
        "median_speedup_fallback_over_native": (
            statistics.median(fallback_per_call) / statistics.median(native_per_call)
        ),
        "paired_bootstrap": boot,
        "correctness": correctness,
        "warmups": {
            "native_elapsed_ns": warmups["native"]["elapsed_ns"],
            "fallback_elapsed_ns": warmups["fallback"]["elapsed_ns"],
        },
        "correctness_ok": correctness_ok,
    }


# ---------------------------------------------------------------------------
# Report + output ownership
# ---------------------------------------------------------------------------


def _non_claims() -> list[str]:
    return [
        "Results apply only to this Mac / recorded machine; no cross-machine claim.",
        "Timed scope is boundary-inclusive product latency through the generated "
        "Rextio wrapper (not internal-only native kernel timing).",
        "No CUDA, MPS, or training/autograd performance claim.",
        "No extrapolation beyond the six frozen cells and recorded pins.",
        "Direct eager and torch.compile are context-only; they never rescue the "
        "Rextio expansion GO gate.",
        "Compilation and first-call warm-up are excluded from steady-state samples.",
        "Smoke runs are not performance evidence.",
    ]


def _write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    gate = result["expansion_go_gate"]
    lines = [
        "# Rextio-torch Phase A product-route benchmark",
        "",
        "Primary rows call the generated Rextio wrapper in two persistent processes; "
        "only `REXTIO_NATIVE_MODE` differs. Context lanes are not Rextio claims.",
        "",
        "## Expansion GO gate",
        "",
        f"- Verdict: **{gate['verdict']}**",
        f"- Aggregate geomean fallback/native speedup: "
        f"`{gate.get('aggregate_geomean_fallback_over_native')}`",
        f"- Aggregate speedup 95% CI: `{gate.get('aggregate_speedup_ci95')}`",
        f"- Target cells with native/fallback CI upper < 1.0: "
        f"`{gate.get('target_cells_ci_upper_below_one')}`",
        "",
    ]
    if gate["reasons"]:
        lines.append("Blocking reasons:")
        lines.extend(f"- `{reason}`" for reason in gate["reasons"])
        lines.append("")
    lines.extend(
        [
            "## Cells",
            "",
            "| Case | Role | Batch | In | Out | Native ms | Fallback ms | "
            "Median n/f | Speedup f/n | n/f CI95 | Eligible |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cell in result["cells"]:
        product = cell["product"]
        boot = product.get("paired_bootstrap") or {}
        ratio_ci = boot.get("ratio_ci95_native_over_fallback") or [None, None]
        lines.append(
            f"| {cell['case_id']} | {cell['role']} | {cell['batch']} | "
            f"{cell['in_features']} | {cell['out_features']} | "
            f"{float(product['native_median_ns_per_call']) / 1e6:.4f} | "
            f"{float(product['fallback_median_ns_per_call']) / 1e6:.4f} | "
            f"{boot.get('median_native_over_fallback')} | "
            f"{boot.get('speedup_geomean_fallback_over_native')} | "
            f"[{ratio_ci[0]}, {ratio_ci[1]}] | {cell.get('primary_eligible')} |"
        )
    lines.extend(["", "## Explicit non-claims", ""])
    lines.extend(f"- {item}" for item in result["non_claims"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _pop_report_evidence(result: dict[str, Any]) -> dict[str, str]:
    provenance = result.get("provenance", {})
    return provenance.pop("report_evidence_files", {})


def _is_authoritative_path(path: Path) -> bool:
    """True when *path* is a harness-owned authoritative result location."""
    resolved = path.resolve()
    if resolved in {FULL_RESULT.resolve(), FULL_REPORT.resolve()}:
        return True
    evidence = EVIDENCE_DIR.resolve()
    try:
        if resolved == evidence or resolved.is_relative_to(evidence):
            return True
    except (ValueError, OSError):
        pass
    # Prefix match covers the case where evidence/ does not exist yet.
    evidence_prefix = str(evidence) + os.sep
    return str(resolved) == str(evidence) or str(resolved).startswith(evidence_prefix)


def _write_smoke_output(result: dict[str, Any], output: Path | None) -> Path:
    """Write a smoke result only to an ignored/temp location."""
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="rextio-torch-smoke-")) / "smoke.json"
    output = output.resolve()
    _require(
        not _is_authoritative_path(output),
        f"smoke output {output} must not touch harness-owned authoritative results",
    )
    _pop_report_evidence(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _write_full_output(result: dict[str, Any]) -> Path:
    """Atomically replace only harness-owned authoritative result + report + evidence."""
    evidence = _pop_report_evidence(result)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if EVIDENCE_DIR.exists():
        shutil.rmtree(EVIDENCE_DIR)
    EVIDENCE_DIR.mkdir(parents=True)
    for name, body in evidence.items():
        (EVIDENCE_DIR / name).write_text(body, encoding="utf-8")

    report_path = FULL_REPORT
    _write_markdown(report_path, result)

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    tmp = FULL_RESULT.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, FULL_RESULT)
    return FULL_RESULT


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------


def _verify_route_and_build(
    project_root: Path,
) -> tuple[dict[str, Any], bytes, bytes, dict[str, Any]]:
    reports_dir = project_root / ".rextio" / "reports"
    check_path = reports_dir / "check.json"
    build_path = reports_dir / "build.json"
    _require(check_path.exists(), "check.json is missing")
    _require(build_path.exists(), "build.json is missing")
    check_bytes = check_path.read_bytes()
    build_bytes = build_path.read_bytes()
    check = json.loads(check_bytes.decode("utf-8"))
    build = json.loads(build_bytes.decode("utf-8"))

    functions = {
        function["qualname"]: function
        for module in check["modules"]
        for function in module["functions"]
    }
    record = functions.get(FUNCTION_QUALNAME)
    _require(record is not None, f"{FUNCTION_QUALNAME} missing from check.json")
    _require(
        record["route"] == EXPECTED_ROUTE,
        f"route is {record['route']!r}, expected {EXPECTED_ROUTE!r}",
    )
    _require(
        record["native_status"] == "accepted",
        f"native_status is {record['native_status']!r}, expected 'accepted'",
    )
    claim_rules = {claim["rule_id"] for claim in (record.get("plugin_claims") or [])}
    _require(
        claim_rules == EXPECTED_CLAIM_RULES,
        f"plugin claims {claim_rules} != {EXPECTED_CLAIM_RULES}",
    )

    native_build = build.get("native_build") or {}
    _require(build.get("status") == "built", "build status is not 'built'")
    _require(native_build.get("status") == "built", "native build did not complete")
    _require(
        build.get("rejected_native_count", 0) == 0,
        "build report rejected a native function",
    )
    native_artifact = Path(native_build["installed_path"]).resolve()
    _require(native_artifact.exists(), f"native artifact missing: {native_artifact}")
    _require(
        native_artifact.is_relative_to(project_root.resolve()),
        "native artifact resolves outside the freshly built project",
    )

    route_evidence = {
        "check_json_sha256": hashlib.sha256(check_bytes).hexdigest(),
        "build_json_sha256": hashlib.sha256(build_bytes).hexdigest(),
        "function": {
            "qualname": FUNCTION_QUALNAME,
            "route": record["route"],
            "native_status": record["native_status"],
            "rule_ids": sorted(claim_rules),
        },
        "native_artifact": {
            "path": str(native_artifact),
            "sha256": _sha256_file(native_artifact),
        },
        "build_status": build.get("status"),
        "native_build_status": native_build.get("status"),
        "accepted_native_count": build.get("accepted_native_count"),
        "rejected_native_count": build.get("rejected_native_count"),
        "installed_path": native_build.get("installed_path"),
    }
    return route_evidence, check_bytes, build_bytes, native_build


def _bench(args: argparse.Namespace) -> dict[str, Any]:
    # Import torch in the parent before any build/extension load.
    _import_torch_configured()
    provenance = _preflight()
    build_env = _configure_libtorch_env()
    # Merge into process env for certification build helpers that inherit os.environ.
    os.environ.update(
        {
            "VIRTUAL_ENV": build_env["VIRTUAL_ENV"],
            "PATH": build_env["PATH"],
            "PYO3_PYTHON": build_env["PYO3_PYTHON"],
            "LIBTORCH_USE_PYTORCH": "1",
        }
    )
    os.environ.pop("LIBTORCH_BYPASS_VERSION_CHECK", None)

    repetitions = 3 if args.smoke else args.repetitions
    target_ns = int(args.target_ms * 1_000_000)
    bootstrap_resamples = 200 if args.smoke else args.bootstrap_resamples
    cases = frozen_cases()
    if args.smoke:
        # Smoke: single smallest target cell only (harness/route gate, not evidence).
        cases = [get_case_for_smoke()]

    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="rextio-torch-bench-") as directory:
        project_root = Path(directory).resolve()
        _write_project(project_root)
        from rextio.plugins.testing import build_certification_project

        compile_started = time.perf_counter_ns()
        project = build_certification_project(project_root)
        compile_ns = time.perf_counter_ns() - compile_started

        route_evidence, check_bytes, build_bytes, native_build = _verify_route_and_build(
            project_root
        )
        provenance["route_evidence"] = route_evidence
        provenance["compile_ns"] = compile_ns
        provenance["report_evidence_files"] = {
            "check.json": check_bytes.decode("utf-8"),
            "build.json": build_bytes.decode("utf-8"),
        }

        build_python = Path(project.build_python_dir).resolve()
        # Fail-closed: the extension staged for workers must be byte-identical to
        # the build report's installed_path (not a cache/global copy).
        staged_natives = sorted(build_python.glob("_rextio_native*"))
        _require(
            len(staged_natives) >= 1,
            f"no _rextio_native artifact under build python dir {build_python}",
        )
        bound = _require_bound_native_artifact(
            str(native_build["installed_path"]),
            str(staged_natives[0]),
        )
        provenance["route_evidence"]["native_artifact"]["timing_path"] = str(bound)
        provenance["route_evidence"]["native_artifact"]["timing_matches_build"] = True

        workers = {
            mode: ModeWorker(mode, build_python, build_env) for mode in ("native", "fallback")
        }
        cells: list[dict[str, Any]] = []
        try:
            # Confirm the native worker can load and pass dtype/device checks.
            probe = workers["native"].run(cases[0], 1)
            _require(
                probe["correctness"]["dtype_ok"],
                "native worker correctness probe failed dtype check",
            )
            for index, case in enumerate(cases):
                product = _measure_product_cell(
                    case,
                    workers,
                    repetitions=repetitions,
                    target_ns=max(FLOOR_NS, target_ns),
                    seed=args.seed + index,
                    bootstrap_resamples=bootstrap_resamples,
                )
                contexts = _context_lanes(
                    case,
                    repetitions=repetitions,
                    target_ns=max(FLOOR_NS, target_ns),
                    seed=args.seed + 100 + index,
                )
                cells.append(
                    {
                        "case_id": case.case_id,
                        "role": case.role,
                        "batch": case.batch,
                        "in_features": case.in_features,
                        "out_features": case.out_features,
                        "seed": case.seed,
                        "product": product,
                        "contexts": contexts,
                        "route_verified": True,
                        "correctness_ok": product["correctness_ok"],
                        "primary_eligible": False,
                        "primary_ineligible_reasons": ["not-yet-evaluated"],
                    }
                )
        finally:
            for worker in workers.values():
                worker.close()

    for cell in cells:
        ok, reasons = _cell_primary_eligible(cell, repetitions=repetitions)
        cell["primary_eligible"] = ok
        cell["primary_ineligible_reasons"] = reasons
        # Surface product-level flag used by gate.
        cell["correctness_ok"] = bool(cell["product"].get("correctness_ok"))

    gate = compute_expansion_go_gate(
        cells,
        repetitions=repetitions,
        bootstrap_resamples=bootstrap_resamples,
        seed=args.seed,
    )

    return {
        "schema": "rextio-torch-phase-a-product-benchmark-v1",
        "smoke": bool(args.smoke),
        "generated_at_unix_ns": time.time_ns(),
        "elapsed_ns": time.perf_counter_ns() - started,
        "configuration": {
            "cases": [case.case_id for case in cases],
            "target_case_ids": list(TARGET_CASE_IDS),
            "context_case_ids": list(CONTEXT_CASE_IDS),
            "repetitions": repetitions,
            "minimum_sample_ms": FLOOR_NS / 1e6,
            "calibration_target_ms": max(DEFAULT_TARGET_MS, args.target_ms),
            "bootstrap_resamples": bootstrap_resamples,
            "seed": args.seed,
            "gc_disabled_during_samples": True,
            "counterbalanced_pairs": True,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "persistent_processes": 2,
            "modes": ["native", "fallback"],
            "context_lanes": ["eager", "torch_compile_fullgraph"],
        },
        "provenance": provenance,
        "cells": cells,
        "expansion_go_gate": gate,
        "non_claims": _non_claims(),
    }


def get_case_for_smoke() -> BenchmarkCase:
    """Return t1 only — smoke is a harness/route check, not performance evidence."""
    return get_case("t1")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: worker mode or parent orchestration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("native", "fallback"))
    parser.add_argument("--build-python")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--target-ms", type=float, default=DEFAULT_TARGET_MS)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Smoke-only explicit output path; ignored for full runs.",
    )
    args = parser.parse_args(argv)

    if args.worker:
        if args.mode is None or args.build_python is None:
            raise SystemExit("worker requires --mode and --build-python")
        return _worker_main(args)

    result = _bench(args)
    if args.smoke:
        output = _write_smoke_output(result, args.output)
    else:
        _require(
            args.output is None,
            "full runs always write the tracked authoritative result; --output is smoke-only",
        )
        output = _write_full_output(result)
    print(output)
    print(f"expansion_go_gate: {result['expansion_go_gate']['verdict']}")
    return 0


if __name__ == "__main__":
    # Allow ``python benchmarks/bench_phase_a.py`` as a script (workers use the same path).
    if __package__ is None or __package__ == "":
        sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
