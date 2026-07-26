"""Diagnostic small-batch scoring harness (0.1.3 candidate).

Validates full logits and probabilities with numeric tolerance before exact
labels. Diagnostic lanes:

* **native_rextio** — primary measured native lane when an already-built
  project is supplied (no rebuild inside timed samples)
* **eager** / **torch.inference_mode** — context lanes only

All timings are labeled ``diagnostic`` only. Never writes into Phase A/B
authoritative result trees and never claims a performance cohort or speedup.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Final

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.small_batch_scoring_cases import (  # noqa: E402
    CONTROL_FLOW_ROUNDS,
    DEFAULT_TIMING_SAMPLES,
    EXPECTED_NATIVE_ROUTE,
    FROZEN_CASES,
    KERNEL_SOURCE,
    LOGITS_ATOL,
    LOGITS_RTOL,
    PROBS_ATOL,
    PROBS_RTOL,
    SCORE_LABELS_QUALNAME,
    SCORE_LOGITS_QUALNAME,
    SCORE_MODULE_NAME,
    SCORE_PROBS_QUALNAME,
    SmallBatchScoringCase,
    frozen_cases,
)
from rextio_torch.invocation_scope import (  # noqa: E402
    INVOCATION_SCOPE_OPTIMIZATION_ACTIVE,
    inspect_core_function_scope_capability,
    production_no_grad_baseline_summary,
)

ROOT = _ROOT
RESULTS_PHASE_A = ROOT / "benchmarks" / "results"
RESULTS_PHASE_B = ROOT / "benchmarks" / "results_phase_b"
HARNESS_KIND = "diagnostic-small-batch-scoring"
HARNESS_VERSION = "0.1.3-candidate"
NATIVE_UNAVAILABLE_REASON = (
    "native Rextio lane not measured: supply an already-built project via "
    "--built-project or load_native_scoring_lane(...); the ordinary CLI does "
    "not build or imply native measurement"
)
NATIVE_EXTENSION_MODULE: Final[str] = "_rextio_native"
# Sentinel: REXTIO_NATIVE_MODE was unset in the process environment.
_MODE_UNSET: Final[object] = object()


def _require(condition: object, message: str) -> None:
    """Fail-closed runtime gate that survives ``python -O``."""
    if not condition:
        raise RuntimeError(f"small-batch scoring diagnostic failed: {message}")


def require_positive_timing_samples(timing_samples: int) -> int:
    """Validate diagnostic timing sample count is a positive integer."""
    if not isinstance(timing_samples, int) or isinstance(timing_samples, bool):
        raise RuntimeError(
            "small-batch scoring diagnostic failed: timing_samples must be a "
            f"positive int, got {timing_samples!r}"
        )
    if timing_samples < 1:
        raise RuntimeError(
            "small-batch scoring diagnostic failed: timing_samples must be >= 1, "
            f"got {timing_samples}"
        )
    return timing_samples


def _is_historical_authoritative_path(path: Path) -> bool:
    """Return whether ``path`` sits under Phase A/B authoritative trees."""
    resolved = path.resolve()
    for root in (RESULTS_PHASE_A, RESULTS_PHASE_B):
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def refuse_historical_output(path: Path) -> None:
    """Refuse writes that would touch Phase A/B recorded results."""
    if _is_historical_authoritative_path(path):
        raise RuntimeError(
            "refusing to write diagnostic small-batch scoring output under "
            f"historical benchmark tree: {path}"
        )


def eager_score_logits(
    torch: Any,
    x: Any,
    mean: Any,
    scale: Any,
    hidden_weight: Any,
    hidden_bias: Any,
    class_weight: Any,
    class_bias: Any,
    rounds: int,
    phase: int,
) -> Any:
    """Eager reference for the full pipeline through classification linear."""
    functional = torch.nn.functional
    centered = x - mean
    normalized = centered / scale
    hidden = functional.linear(normalized, hidden_weight, hidden_bias)
    for layer in range(rounds):
        if (layer + phase) % 2 == 0:
            hidden = hidden.relu()
        else:
            hidden = hidden.tanh()
    return functional.linear(hidden, class_weight, class_bias)


def eager_score_probabilities(
    torch: Any,
    x: Any,
    mean: Any,
    scale: Any,
    hidden_weight: Any,
    hidden_bias: Any,
    class_weight: Any,
    class_bias: Any,
    rounds: int,
    phase: int,
) -> Any:
    """Eager reference through softmax."""
    logits = eager_score_logits(
        torch,
        x,
        mean,
        scale,
        hidden_weight,
        hidden_bias,
        class_weight,
        class_bias,
        rounds,
        phase,
    )
    return logits.softmax(dim=1)


def eager_score_labels(
    torch: Any,
    x: Any,
    mean: Any,
    scale: Any,
    hidden_weight: Any,
    hidden_bias: Any,
    class_weight: Any,
    class_bias: Any,
    rounds: int,
    phase: int,
) -> Any:
    """Eager reference through argmax labels."""
    probabilities = eager_score_probabilities(
        torch,
        x,
        mean,
        scale,
        hidden_weight,
        hidden_bias,
        class_weight,
        class_bias,
        rounds,
        phase,
    )
    return probabilities.argmax(dim=1, keepdim=False)


def make_case_tensors(torch: Any, case: SmallBatchScoringCase) -> dict[str, Any]:
    """Build deterministic float32 CPU tensors for one cell."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)
    x = torch.randn(case.x_shape, dtype=torch.float32, generator=generator)
    mean = torch.randn(case.mean_shape, dtype=torch.float32, generator=generator)
    # Positive scales avoid division-by-zero pathologies in the diagnostic.
    scale = (
        torch.rand(case.scale_shape, dtype=torch.float32, generator=generator) * 0.5 + 0.5
    )
    hidden_weight = torch.randn(
        case.hidden_weight_shape, dtype=torch.float32, generator=generator
    )
    hidden_bias = torch.randn(
        case.hidden_bias_shape, dtype=torch.float32, generator=generator
    )
    class_weight = torch.randn(
        case.class_weight_shape, dtype=torch.float32, generator=generator
    )
    class_bias = torch.randn(
        case.class_bias_shape, dtype=torch.float32, generator=generator
    )
    return {
        "x": x,
        "mean": mean,
        "scale": scale,
        "hidden_weight": hidden_weight,
        "hidden_bias": hidden_bias,
        "class_weight": class_weight,
        "class_bias": class_bias,
        "rounds": case.rounds,
        "phase": case.phase,
    }


def pipeline_args(tensors: Mapping[str, Any]) -> tuple[Any, ...]:
    """Positional argument tuple shared by eager and native callables."""
    return (
        tensors["x"],
        tensors["mean"],
        tensors["scale"],
        tensors["hidden_weight"],
        tensors["hidden_bias"],
        tensors["class_weight"],
        tensors["class_bias"],
        tensors["rounds"],
        tensors["phase"],
    )


def _allclose(torch: Any, actual: Any, expected: Any, *, rtol: float, atol: float) -> bool:
    if type(actual) is not torch.Tensor or type(expected) is not torch.Tensor:
        return False
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return False
    if actual.device != expected.device:
        return False
    return bool(torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))


def validate_pipeline_outputs(
    torch: Any,
    *,
    logits: Any,
    probabilities: Any,
    labels: Any,
    expected_logits: Any,
    expected_probabilities: Any,
    expected_labels: Any,
) -> dict[str, object]:
    """Validate logits and probabilities before exact labels.

    Raises ``RuntimeError`` on the first failed stage so argmax cannot hide
    drift when callers only inspect a final boolean.
    """
    logits_ok = _allclose(
        torch, logits, expected_logits, rtol=LOGITS_RTOL, atol=LOGITS_ATOL
    )
    _require(logits_ok, "full logits diverged beyond numeric tolerance")
    probs_ok = _allclose(
        torch,
        probabilities,
        expected_probabilities,
        rtol=PROBS_RTOL,
        atol=PROBS_ATOL,
    )
    _require(probs_ok, "full probabilities diverged beyond numeric tolerance")
    labels_ok = type(labels) is torch.Tensor and bool(
        torch.equal(labels, expected_labels)
    )
    _require(labels_ok, "exact labels diverged after logits/probabilities matched")
    return {
        "logits_ok": True,
        "probabilities_ok": True,
        "labels_ok": True,
        "validation_order": ["logits", "probabilities", "labels"],
    }


def _call_pipeline(
    fn_logits: Callable[..., Any],
    fn_probs: Callable[..., Any],
    fn_labels: Callable[..., Any],
    tensors: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    args = pipeline_args(tensors)
    return fn_logits(*args), fn_probs(*args), fn_labels(*args)


def _time_callable(
    fn: Callable[[], Any],
    *,
    timing_samples: int,
    warmup: int = 1,
) -> dict[str, Any]:
    samples = require_positive_timing_samples(timing_samples)
    for _ in range(warmup):
        fn()
    samples_ns: list[int] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        fn()
        samples_ns.append(time.perf_counter_ns() - start)
    return {
        "timing_class": "diagnostic",
        "timing_samples": samples,
        "samples_ns": samples_ns,
        "median_ns": int(statistics.median(samples_ns)),
        "mean_ns": float(statistics.fmean(samples_ns)),
        "speedup_claim": None,
        "official_cohort": False,
        "measured": True,
        "available": True,
    }


def unavailable_native_timing(*, reason: str = NATIVE_UNAVAILABLE_REASON) -> dict[str, Any]:
    """Explicit native omission record (never implies native was measured)."""
    return {
        "timing_class": "diagnostic",
        "lane": "native_rextio",
        "available": False,
        "measured": False,
        "reason": reason,
        "speedup_claim": None,
        "official_cohort": False,
    }


def _package_root_of(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _evict_lane_modules(package_root: str) -> None:
    """Drop dedicated package and top-level native extension from ``sys.modules``."""
    for name in list(sys.modules):
        if name == package_root or name.startswith(package_root + "."):
            sys.modules.pop(name, None)
        if name == NATIVE_EXTENSION_MODULE or name.startswith(
            NATIVE_EXTENSION_MODULE + "."
        ):
            sys.modules.pop(name, None)


def _snapshot_native_mode() -> object:
    if "REXTIO_NATIVE_MODE" in os.environ:
        return os.environ["REXTIO_NATIVE_MODE"]
    return _MODE_UNSET


def _restore_native_mode(prior: object) -> None:
    if prior is _MODE_UNSET:
        os.environ.pop("REXTIO_NATIVE_MODE", None)
    else:
        os.environ["REXTIO_NATIVE_MODE"] = str(prior)


def _force_build_dir_to_sys_path_front(build_dir_str: str) -> None:
    """Place the selected build tree at ``sys.path[0]``, removing other copies."""
    sys.path[:] = [entry for entry in sys.path if entry != build_dir_str]
    sys.path.insert(0, build_dir_str)


def _path_contained_in_build(path: Path, build_dir: Path) -> bool:
    try:
        path.resolve().relative_to(build_dir.resolve())
        return True
    except ValueError:
        return False


def _require_native_extension_in_build(build_dir: Path) -> str:
    """Fail closed unless ``_rextio_native`` is loaded from the selected build tree."""
    native_mod = sys.modules.get(NATIVE_EXTENSION_MODULE)
    _require(
        native_mod is not None,
        f"{NATIVE_EXTENSION_MODULE!r} was not loaded after importing the scoring "
        "module; check.json alone is not extension provenance",
    )
    assert isinstance(native_mod, ModuleType)
    native_file = getattr(native_mod, "__file__", None)
    _require(
        isinstance(native_file, str) and native_file,
        f"{NATIVE_EXTENSION_MODULE!r} has no __file__; cannot verify build provenance",
    )
    resolved = Path(native_file).resolve()
    _require(
        _path_contained_in_build(resolved, build_dir),
        f"{NATIVE_EXTENSION_MODULE!r} loaded from {resolved}, not under {build_dir}",
    )
    return str(resolved)


@dataclass
class NativeScoringLane:
    """Retained native wrappers with explicit process-global isolation.

    Loaded via :func:`load_native_scoring_lane`. While active, may keep
    ``REXTIO_NATIVE_MODE=native`` and the build tree at ``sys.path[0]`` for
    lazy imports. Callers **must** ``close()`` (or use ``with``) after
    measurement so another generated project cannot reuse the extension.
    """

    project_root: Path
    build_python_dir: Path
    module_name: str
    score_logits: Callable[..., Any]
    score_probabilities: Callable[..., Any]
    score_labels: Callable[..., Any]
    routes: dict[str, dict[str, Any]]
    native_extension_path: str
    package_root: str
    _prior_native_mode: object
    _prior_sys_path: list[str]
    available: bool = True
    measured_capable: bool = True
    process_state_owned: bool = True
    _closed: bool = field(default=False, repr=False)

    @property
    def closed(self) -> bool:
        """Return whether :meth:`close` has already run."""
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("NativeScoringLane is closed")

    def close(self) -> None:
        """Restore process globals and evict dedicated modules (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if not self.process_state_owned:
            return
        sys.path[:] = list(self._prior_sys_path)
        _restore_native_mode(self._prior_native_mode)
        _evict_lane_modules(self.package_root)

    def __enter__(self) -> NativeScoringLane:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.close()
        return False

    def to_report_dict(self) -> dict[str, Any]:
        """JSON-ready native lane provenance (includes extension path)."""
        return {
            "available": self.available and not self._closed,
            "measured_capable": self.measured_capable,
            "project_root": str(self.project_root),
            "build_python_dir": str(self.build_python_dir),
            "module_name": self.module_name,
            "package_root": self.package_root,
            "routes": self.routes,
            "expected_route": EXPECTED_NATIVE_ROUTE,
            "native_extension_module": NATIVE_EXTENSION_MODULE,
            "native_extension_path": self.native_extension_path,
            "rebuild_inside_timing": False,
            "closed": self._closed,
            "process_state_owned": self.process_state_owned,
        }


def make_synthetic_native_lane(
    *,
    score_logits: Callable[..., Any],
    score_probabilities: Callable[..., Any],
    score_labels: Callable[..., Any],
    routes: dict[str, dict[str, Any]] | None = None,
    native_extension_path: str = "/synthetic/_rextio_native.so",
    project_root: Path | None = None,
    build_python_dir: Path | None = None,
    module_name: str = SCORE_MODULE_NAME,
) -> NativeScoringLane:
    """Build a non-owning lane for pure harness unit tests (no process mutation)."""
    root = project_root or Path("/synthetic/built-project")
    build = build_python_dir or (root / ".rextio" / "build" / "python")
    default_routes = {
        SCORE_LOGITS_QUALNAME: {
            "native_status": "accepted",
            "route": EXPECTED_NATIVE_ROUTE,
        },
        SCORE_PROBS_QUALNAME: {
            "native_status": "accepted",
            "route": EXPECTED_NATIVE_ROUTE,
        },
        SCORE_LABELS_QUALNAME: {
            "native_status": "accepted",
            "route": EXPECTED_NATIVE_ROUTE,
        },
    }
    return NativeScoringLane(
        project_root=root,
        build_python_dir=build,
        module_name=module_name,
        score_logits=score_logits,
        score_probabilities=score_probabilities,
        score_labels=score_labels,
        routes=routes if routes is not None else default_routes,
        native_extension_path=native_extension_path,
        package_root=_package_root_of(module_name),
        _prior_native_mode=_MODE_UNSET,
        _prior_sys_path=list(sys.path),
        process_state_owned=False,
    )


def _project_root_of(project: Any) -> Path:
    if hasattr(project, "project_root"):
        return Path(project.project_root)
    return Path(project)


def verify_native_routes(
    project_root: Path,
    *,
    qualnames: tuple[str, str, str] = (
        SCORE_LOGITS_QUALNAME,
        SCORE_PROBS_QUALNAME,
        SCORE_LABELS_QUALNAME,
    ),
) -> dict[str, dict[str, Any]]:
    """Fail closed unless every scoring entry is native-plugin accepted.

    Route checks are necessary but **not** sufficient provenance; the loaded
    ``_rextio_native`` extension path must still be verified separately.
    """
    check_path = project_root / ".rextio" / "reports" / "check.json"
    _require(check_path.is_file(), f"missing check report at {check_path}")
    report = json.loads(check_path.read_text(encoding="utf-8"))
    by_qualname: dict[str, dict[str, Any]] = {}
    for module in report.get("modules", ()):
        for function in module.get("functions", ()):
            qualname = function.get("qualname")
            if isinstance(qualname, str):
                by_qualname[qualname] = function
    routes: dict[str, dict[str, Any]] = {}
    for qualname in qualnames:
        record = by_qualname.get(qualname)
        _require(record is not None, f"{qualname} missing from check.json")
        assert record is not None
        route = str(record.get("route", ""))
        status = record.get("native_status")
        _require(
            status == "accepted" and route == EXPECTED_NATIVE_ROUTE,
            f"{qualname} not native-plugin accepted "
            f"(native_status={status!r}, route={route!r})",
        )
        routes[qualname] = {
            "native_status": status,
            "route": route,
        }
    return routes


def load_native_scoring_lane(
    project: Any,
    *,
    module_name: str = SCORE_MODULE_NAME,
    qualnames: tuple[str, str, str] | None = None,
) -> NativeScoringLane:
    """Import and retain native wrappers once from an already-built tree.

    Accepts a :class:`~rextio.plugins.testing.CertifiedProject` or a project
    root path that already contains ``.rextio/build/python`` and reports.
    Does **not** rebuild.

    Process isolation:

    * snapshot ``REXTIO_NATIVE_MODE`` and ``sys.path``
    * force the selected build Python directory to ``sys.path[0]``
    * evict dedicated package and ``_rextio_native`` before import
    * on any failure, restore env/path and evict partial modules
    * after import, require ``_rextio_native.__file__`` under the build tree
    * caller must :meth:`NativeScoringLane.close` (or ``with``) after use
    """
    resolved_qualnames = qualnames or (
        SCORE_LOGITS_QUALNAME,
        SCORE_PROBS_QUALNAME,
        SCORE_LABELS_QUALNAME,
    )
    project_root = _project_root_of(project).resolve()
    if hasattr(project, "build_python_dir"):
        build_dir = Path(project.build_python_dir).resolve()
    else:
        build_dir = (project_root / ".rextio" / "build" / "python").resolve()
    _require(build_dir.is_dir(), f"generated Python build tree missing: {build_dir}")
    routes = verify_native_routes(project_root, qualnames=resolved_qualnames)

    package_root = _package_root_of(module_name)
    prior_mode = _snapshot_native_mode()
    prior_sys_path = list(sys.path)
    build_dir_str = str(build_dir)
    # Any failure after these snapshots must restore env/path and evict modules,
    # including failures inside path-front mutation or NativeScoringLane construction.
    try:
        os.environ["REXTIO_NATIVE_MODE"] = "native"
        _force_build_dir_to_sys_path_front(build_dir_str)
        _evict_lane_modules(package_root)
        module = importlib.import_module(module_name)
        loaded_from = getattr(module, "__file__", None)
        _require(
            isinstance(loaded_from, str)
            and _path_contained_in_build(Path(loaded_from), build_dir),
            f"module {module_name!r} loaded from {loaded_from!r}, not {build_dir}",
        )
        extension_path = _require_native_extension_in_build(build_dir)
        score_logits = getattr(module, "score_logits")
        score_probabilities = getattr(module, "score_probabilities")
        score_labels = getattr(module, "score_labels")
        # Active lane retains native mode + build path for lazy imports until close.
        return NativeScoringLane(
            project_root=project_root,
            build_python_dir=build_dir,
            module_name=module_name,
            score_logits=score_logits,
            score_probabilities=score_probabilities,
            score_labels=score_labels,
            routes=routes,
            native_extension_path=extension_path,
            package_root=package_root,
            _prior_native_mode=prior_mode,
            _prior_sys_path=prior_sys_path,
            process_state_owned=True,
        )
    except Exception:
        sys.path[:] = list(prior_sys_path)
        _restore_native_mode(prior_mode)
        _evict_lane_modules(package_root)
        raise


def run_case_diagnostic(
    torch: Any,
    case: SmallBatchScoringCase,
    *,
    timing_samples: int = DEFAULT_TIMING_SAMPLES,
    native_lane: NativeScoringLane | None = None,
) -> dict[str, Any]:
    """Run correctness and diagnostic timings for one predeclared cell."""
    samples = require_positive_timing_samples(timing_samples)
    tensors = make_case_tensors(torch, case)
    expected_logits = eager_score_logits(torch, **tensors)
    expected_probs = eager_score_probabilities(torch, **tensors)
    expected_labels = eager_score_labels(torch, **tensors)

    def eager_logits_fn(*args: Any) -> Any:
        return eager_score_logits(torch, *args)

    def eager_probs_fn(*args: Any) -> Any:
        return eager_score_probabilities(torch, *args)

    def eager_labels_fn(*args: Any) -> Any:
        return eager_score_labels(torch, *args)

    logits, probs, labels = _call_pipeline(
        eager_logits_fn, eager_probs_fn, eager_labels_fn, tensors
    )
    correctness: dict[str, Any] = {
        "eager": validate_pipeline_outputs(
            torch,
            logits=logits,
            probabilities=probs,
            labels=labels,
            expected_logits=expected_logits,
            expected_probabilities=expected_probs,
            expected_labels=expected_labels,
        )
    }

    def eager_once() -> Any:
        return eager_score_labels(torch, **tensors)

    timings: dict[str, Any] = {
        "eager": {
            **_time_callable(eager_once, timing_samples=samples),
            "lane": "eager",
            "role": "context",
        },
    }

    inference_mode = getattr(torch, "inference_mode", None)
    if callable(inference_mode):
        def inference_once() -> Any:
            with torch.inference_mode():
                return eager_score_labels(torch, **tensors)

        with torch.inference_mode():
            im_logits, im_probs, im_labels = _call_pipeline(
                eager_logits_fn, eager_probs_fn, eager_labels_fn, tensors
            )
        correctness["torch_inference_mode"] = validate_pipeline_outputs(
            torch,
            logits=im_logits,
            probabilities=im_probs,
            labels=im_labels,
            expected_logits=expected_logits,
            expected_probabilities=expected_probs,
            expected_labels=expected_labels,
        )
        timings["torch_inference_mode"] = {
            **_time_callable(inference_once, timing_samples=samples),
            "lane": "torch_inference_mode",
            "role": "context",
        }
    else:
        timings["torch_inference_mode"] = {
            "timing_class": "diagnostic",
            "lane": "torch_inference_mode",
            "role": "context",
            "available": False,
            "measured": False,
            "reason": "torch.inference_mode unavailable",
            "speedup_claim": None,
            "official_cohort": False,
        }

    if native_lane is None:
        timings["native_rextio"] = unavailable_native_timing()
        correctness["native_rextio"] = {
            "available": False,
            "measured": False,
            "reason": NATIVE_UNAVAILABLE_REASON,
        }
    else:
        if native_lane.closed:
            raise RuntimeError(
                "small-batch scoring diagnostic failed: NativeScoringLane is closed"
            )
        args = pipeline_args(tensors)
        native_logits = native_lane.score_logits(*args)
        native_probs = native_lane.score_probabilities(*args)
        native_labels = native_lane.score_labels(*args)
        correctness["native_rextio"] = {
            **validate_pipeline_outputs(
                torch,
                logits=native_logits,
                probabilities=native_probs,
                labels=native_labels,
                expected_logits=expected_logits,
                expected_probabilities=expected_probs,
                expected_labels=expected_labels,
            ),
            "available": True,
            "measured": True,
        }

        def native_once() -> Any:
            return native_lane.score_labels(*args)

        timings["native_rextio"] = {
            **_time_callable(native_once, timing_samples=samples),
            "lane": "native_rextio",
            "role": "primary_diagnostic",
            "rebuild_inside_timing": False,
            "speedup_claim": None,
        }

    return {
        "case": case.to_dict(),
        "control_flow_rounds": case.rounds,
        "timing_samples": samples,
        "timing_class": "diagnostic",
        "official_cohort": False,
        "speedup_claim": None,
        "correctness": correctness,
        "timings": timings,
        "native_lane_available": native_lane is not None,
        "native_measured": native_lane is not None,
        "invocation_scope_optimization_active": INVOCATION_SCOPE_OPTIMIZATION_ACTIVE,
    }


def build_report(
    cells: list[dict[str, Any]],
    *,
    timing_samples: int = DEFAULT_TIMING_SAMPLES,
    native_lane: NativeScoringLane | None = None,
) -> dict[str, Any]:
    """Assemble a diagnostic-only JSON report (no GO gate, no speedup)."""
    samples = require_positive_timing_samples(timing_samples)
    capability = inspect_core_function_scope_capability()
    native_measured = any(bool(cell.get("native_measured")) for cell in cells)
    return {
        "harness": HARNESS_KIND,
        "harness_version": HARNESS_VERSION,
        "timing_class": "diagnostic",
        "official_cohort": False,
        "speedup_claim": None,
        "control_flow_rounds": CONTROL_FLOW_ROUNDS,
        "timing_samples": samples,
        "kernel_present": "score_labels" in KERNEL_SOURCE,
        "production_no_grad_baseline": production_no_grad_baseline_summary(),
        "native_lane": (
            native_lane.to_report_dict()
            if native_lane is not None
            else {
                "available": False,
                "measured_capable": False,
                "measured": False,
                "reason": NATIVE_UNAVAILABLE_REASON,
            }
        ),
        "native_measured": native_measured,
        "core_function_scope": {
            "plugin_api_version": capability.plugin_api_version,
            "has_function_body_scope_hook": capability.has_function_body_scope_hook,
            "invocation_scope_optimization_active": (
                capability.invocation_scope_optimization_active
            ),
            "limitation": capability.limitation,
            "proposal_id": capability.proposal_id,
            "proposal_status": capability.proposal_status,
            "lowered_expr_fields": sorted(capability.lowered_expr_fields),
        },
        "cells": cells,
    }


def write_diagnostic_output(report: Mapping[str, Any], path: Path) -> None:
    """Write JSON only after refusing historical Phase A/B paths."""
    refuse_historical_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_diagnostic_matrix(
    torch: Any,
    *,
    timing_samples: int = DEFAULT_TIMING_SAMPLES,
    native_lane: NativeScoringLane | None = None,
) -> dict[str, Any]:
    """Run all predeclared cells and assemble the diagnostic report."""
    samples = require_positive_timing_samples(timing_samples)
    cells = [
        run_case_diagnostic(
            torch,
            case,
            timing_samples=samples,
            native_lane=native_lane,
        )
        for case in frozen_cases()
    ]
    return build_report(cells, timing_samples=samples, native_lane=native_lane)


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the diagnostic harness."""
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic small-batch scoring pipeline (no official cohort, "
            "no speedup claim). Native Rextio is measured only when "
            "--built-project points at an already-built tree."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the diagnostic and write only to a temp path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (must not be under Phase A/B results).",
    )
    parser.add_argument(
        "--timing-samples",
        type=int,
        default=DEFAULT_TIMING_SAMPLES,
        help=(
            "Positive diagnostic timing sample count per lane "
            f"(default {DEFAULT_TIMING_SAMPLES}; independent of control-flow rounds)."
        ),
    )
    parser.add_argument(
        "--built-project",
        type=Path,
        default=None,
        help=(
            "Already-built project root containing .rextio/build/python and "
            "check.json. Enables the native Rextio diagnostic lane without "
            "rebuilding inside timed samples."
        ),
    )
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"torch is required for the diagnostic harness: {exc}") from exc

    samples = require_positive_timing_samples(args.timing_samples)
    native_lane: NativeScoringLane | None = None
    try:
        if args.built_project is not None:
            native_lane = load_native_scoring_lane(args.built_project)

        report = run_diagnostic_matrix(
            torch,
            timing_samples=samples,
            native_lane=native_lane,
        )
        output = args.output
        if args.smoke or output is None:
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                suffix="-small-batch-scoring-diagnostic.json",
                delete=False,
                encoding="utf-8",
            )
            output = Path(handle.name)
            handle.close()
        write_diagnostic_output(report, output)
        print(
            json.dumps(
                {
                    "timing_class": "diagnostic",
                    "official_cohort": False,
                    "speedup_claim": None,
                    "output": str(output),
                    "cases": [case.case_id for case in FROZEN_CASES],
                    "control_flow_rounds": CONTROL_FLOW_ROUNDS,
                    "timing_samples": samples,
                    "native_measured": report["native_measured"],
                    "native_lane_available": native_lane is not None,
                    "native_extension_path": (
                        native_lane.native_extension_path
                        if native_lane is not None
                        else None
                    ),
                    "invocation_scope_optimization_active": (
                        INVOCATION_SCOPE_OPTIMIZATION_ACTIVE
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        if native_lane is not None:
            native_lane.close()


if __name__ == "__main__":
    # Avoid accidental LIBTORCH bypass if a caller exports it.
    os.environ.pop("LIBTORCH_BYPASS_VERSION_CHECK", None)
    raise SystemExit(main())
