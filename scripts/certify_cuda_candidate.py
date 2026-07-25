"""Manual real-NVIDIA execution candidate for the frozen Torch CUDA E2 slice.

This script is deliberately excluded from ordinary CI.  It creates evidence
only after a real provider probe, Core code generation, native execution,
CUDA-graph replay, transfer checks, and one-wheel libtorch identity checks all
succeed.  The evidence never asserts released support or certification.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType
from typing import Any

from rextio.analyzer.project_scanner import analyze_project
from rextio.build.orchestrator import generate_source_artifact
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.devices import (
    DEVICE_PROVIDER_ENTRY_POINT,
    DeviceProviderOptions,
    DeviceProviderSelection,
)
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec
from rextio.targets.plan import TargetPlan

from rextio_torch.plugin import PLUGIN_ID, plugin
_module_prefix = f"{__package__}." if __package__ else ""
_build_candidate = importlib.import_module(f"{_module_prefix}build_cuda_candidate")
_evidence = importlib.import_module(f"{_module_prefix}verify_cuda_e2_evidence")

CORE_CAPABILITY_ID = _build_candidate.CORE_CAPABILITY_ID
CORE_PROVIDER_ID = _build_candidate.CORE_PROVIDER_ID
E2_RULES = _build_candidate.E2_RULES
KERNELS = _build_candidate.KERNELS
ALLOWED_SMS = _evidence.ALLOWED_SMS
EvidenceError = _evidence.EvidenceError
canonical_json = _evidence.canonical_json
forbidden_profiler_events = _evidence.forbidden_profiler_events
make_envelope = _evidence.make_envelope
parse_ldd = _evidence.parse_ldd
parse_proc_maps = _evidence.parse_proc_maps
redact_path = _evidence.redact_path
sha256_file = _evidence.sha256_file
verify_document = _evidence.verify_document

CORE_COMMIT = "7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0"
PROVIDER_COMMIT = "a5fb427e91710b65f54ee5b8e33706c45840cf9c"
TORCH_BASE = "7d6fb1b606bfab6530a7ff96317081b2fd1c1b22"
RTOL = 1e-5
ATOL = 1e-6
TORCH_GLOBAL_DEPS_BASENAME = _evidence.TORCH_GLOBAL_DEPS_BASENAME


class _PluginEntryPoint:
    name = PLUGIN_ID

    def load(self):
        return plugin


class _ProviderDistribution:
    name = CORE_PROVIDER_ID
    version = "0.1.0"


class _ProviderEntryPoint:
    group = DEVICE_PROVIDER_ENTRY_POINT
    name = CORE_PROVIDER_ID
    value = "rextio_device_cuda.provider:provider"
    dist = _ProviderDistribution()

    def __init__(self, provider: object) -> None:
        self.provider = provider

    def load(self) -> object:
        return self.provider


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(f"command failed ({command[0]}): {tail}")
    return completed.stdout.strip()


def _git(root: Path, *args: str) -> str:
    return _run(["git", *args], cwd=root, timeout=60)


def _require_checkout(root: Path, expected: str, *, ancestor: str | None = None) -> None:
    if _git(root, "rev-parse", "HEAD") != expected:
        raise RuntimeError(f"{root.name} is not at the explicitly expected commit")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError(f"{root.name} tracked checkout is dirty")
    if ancestor is not None:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, expected],
            cwd=root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError("Torch candidate does not descend from the E2 build-only merge")


def _require_module_under_root(module: ModuleType, root: Path, label: str) -> None:
    module_file = module.__file__
    if module_file is None:
        raise RuntimeError(f"{label} module has no filesystem identity")
    try:
        Path(module_file).resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        raise RuntimeError(f"{label} module was not imported from its exact checkout") from None


def _build_probe(provider_root: Path) -> Path:
    _run(
        [
            "cargo",
            "+1.93.1",
            "build",
            "--locked",
            "--release",
            "-p",
            "rextio-cuda-driver-probe",
        ],
        cwd=provider_root,
    )
    probe = (provider_root / "target/release/rextio-cuda-driver-probe").resolve(strict=True)
    if not probe.is_file() or not os.access(probe, os.X_OK):
        raise RuntimeError("reviewed provider probe was not produced")
    return probe


def _write_fixture(root: Path) -> None:
    package = root / "src/cuda_app"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNELS, encoding="utf-8")


def _generate(root: Path, probe: Path, sm: str) -> object:
    from rextio_device_cuda.config import CudaProviderConfig
    from rextio_device_cuda.provider import CudaDeviceProvider

    _write_fixture(root)
    config = RextioConfig()
    registry = load_plugin_registry(
        PluginConfig(enabled=(PLUGIN_ID,)),
        TargetSpec(),
        entry_points=(_PluginEntryPoint(),),
        full_config=config,
    )
    analysis = analyze_project(
        root,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=config,
    )
    [function] = analysis.accepted_native_functions
    if tuple(claim.rule_id for claim in function.plugin_claims) != E2_RULES:
        raise RuntimeError("analyzer did not accept exactly the frozen four-op chain")
    provider = CudaDeviceProvider(CudaProviderConfig())
    result = generate_source_artifact(
        root,
        analysis,
        "cpython",
        target_plan=TargetPlan(TargetSpec(), registry),
        device_selection=DeviceProviderSelection(CORE_PROVIDER_ID, CORE_CAPABILITY_ID),
        device_options=DeviceProviderOptions(
            values=(
                ("probe_executable", str(probe)),
                ("device_ordinal", "0"),
                ("sm", sm),
            )
        ),
        device_entry_points=(_ProviderEntryPoint(provider),),
    )
    if result.native_source.status != "generated":
        raise RuntimeError(f"Core source generation failed: {result.native_source}")
    [plan] = result.device_provider_plans
    if plan["report"]["support_claim"] is not False:
        raise RuntimeError("provider preflight overclaimed support")
    return result


def _discover_extension(release_dir: Path) -> Path:
    """Return exactly one non-empty top-level generated Linux cdylib."""
    candidates = tuple(
        path
        for path in sorted(release_dir.glob("*_rextio_native*.so"))
        if path.parent == release_dir and path.is_file() and path.stat().st_size > 0
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Cargo did not produce exactly one generated Linux cdylib: {candidates}")
    return candidates[0]


def _build_extension(rust_dir: Path, env: dict[str, str]) -> Path:
    _run(
        [
            "cargo",
            "+1.93.1",
            "build",
            "--release",
            "--manifest-path",
            str(rust_dir / "Cargo.toml"),
        ],
        cwd=rust_dir,
        env=env,
    )
    release_dir = rust_dir / "target/release"
    artifact = _discover_extension(release_dir)
    installed = rust_dir.parent.parent / "build/python" / (
        "_rextio_native" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    )
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, installed)
    return installed.resolve(strict=True)


def _bound_cargo_environment(torch: ModuleType) -> dict[str, str]:
    """Bind torch-sys and PyO3 to this exact active CPython 3.11 environment."""
    python = Path(sys.executable).absolute()
    venv = Path(sys.prefix).absolute()
    if not python.is_file():
        raise RuntimeError("running Python executable does not exist")
    if sys.prefix == sys.base_prefix or python.parent != venv / "bin":
        raise RuntimeError("CUDA E2 requires an active CPython 3.11 virtual environment")
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{python.parent}{os.pathsep}{env.get('PATH', '')}"
    env["PYO3_PYTHON"] = str(python)
    env["LIBTORCH_USE_PYTORCH"] = "1"
    env.pop("LIBTORCH_BYPASS_VERSION_CHECK", None)
    probe = _run(
        [
            "python",
            "-c",
            (
                "import json,sys,torch;"
                "print(json.dumps({'executable':sys.executable,"
                "'implementation':sys.implementation.name,"
                "'python':[sys.version_info.major,sys.version_info.minor],"
                "'torch':torch.__version__.split('+')[0]}))"
            ),
        ],
        env=env,
        timeout=60,
    )
    identity = json.loads(probe)
    probe_executable = identity.get("executable")
    if (
        identity.get("implementation") != "cpython"
        or identity.get("python") != [3, 11]
        or identity.get("torch") != "2.11.0"
        or not isinstance(probe_executable, str)
        or not os.path.samefile(probe_executable, python)
        or torch.__version__.split("+")[0] != identity["torch"]
    ):
        raise RuntimeError("Cargo Python/torch identity differs from the running interpreter")
    return env


def _load_extension(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_rextio_native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("generated extension cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_rextio_native"] = module
    spec.loader.exec_module(module)
    return module


def _tensor_snapshot(torch: ModuleType, values: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "value": value.detach().clone(),
            "stride": tuple(value.stride()),
            "storage_offset": value.storage_offset(),
            "data_ptr": value.data_ptr(),
        }
        for value in values
    )


def _assert_inputs_unchanged(
    torch: ModuleType, values: tuple[Any, ...], snapshots: tuple[dict[str, Any], ...]
) -> None:
    for value, snapshot in zip(values, snapshots, strict=True):
        if (
            not torch.equal(value, snapshot["value"])
            or tuple(value.stride()) != snapshot["stride"]
            or value.storage_offset() != snapshot["storage_offset"]
            or value.data_ptr() != snapshot["data_ptr"]
        ):
            raise RuntimeError("native call mutated or replaced an input")


def _call_case(torch: ModuleType, function: Any, *, contiguous: bool) -> tuple[float, float, float]:
    torch.manual_seed(1701 if contiguous else 1702)
    if contiguous:
        x = torch.randn(4, 3, device="cuda:0", dtype=torch.float32)
        weight = torch.randn(3, 2, device="cuda:0", dtype=torch.float32)
        bias = torch.randn(2, device="cuda:0", dtype=torch.float32)
    else:
        x = torch.randn(3, 4, device="cuda:0", dtype=torch.float32).transpose(0, 1)
        weight = torch.randn(2, 3, device="cuda:0", dtype=torch.float32).transpose(0, 1)
        bias = torch.randn(4, device="cuda:0", dtype=torch.float32)[::2]
        if x.is_contiguous() or weight.is_contiguous() or bias.is_contiguous():
            raise RuntimeError("noncontiguous fixture was normalized before the boundary")
    values = (x, weight, bias)
    snapshots = _tensor_snapshot(torch, values)
    eager = torch.relu(x @ weight + bias).mean(dim=1)
    native = function(x, weight, bias)
    torch.cuda.synchronize()
    if (
        type(native) is not torch.Tensor
        or str(native.device) != "cuda:0"
        or native.dtype is not torch.float32
        or native.dim() != 1
        or native.layout is not torch.strided
        or native.requires_grad
    ):
        raise RuntimeError("native output violated the frozen output contract")
    if not torch.allclose(native, eager, rtol=RTOL, atol=ATOL, equal_nan=True):
        raise RuntimeError("native output differs from PyTorch eager")
    _assert_inputs_unchanged(torch, values, snapshots)
    difference = (native - eager).abs()
    max_abs = float(difference.max().item())
    denominator = eager.abs().clamp_min(ATOL)
    max_rel = float((difference / denominator).max().item())
    scaled = difference / (ATOL + RTOL * eager.abs())
    max_scaled = float(scaled.max().item())
    held = native
    expected_lifetime = eager.detach().clone()
    del native, eager, x, weight, bias, values, snapshots
    gc.collect()
    if (
        str(held.device) != "cuda:0"
        or held.numel() != 4
        or not torch.allclose(
            held,
            expected_lifetime,
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
        )
        or not math.isfinite(float(held.sum().item()))
    ):
        raise RuntimeError("native output did not outlive its inputs")
    return max_abs, max_rel, max_scaled


def _requires_grad_and_boundary_checks(torch: ModuleType, function: Any) -> None:
    x = torch.randn(4, 3, device="cuda:0", dtype=torch.float32)
    weight = torch.randn(3, 2, device="cuda:0", dtype=torch.float32)
    bias = torch.randn(2, device="cuda:0", dtype=torch.float32)
    grad_values = tuple(value.detach().clone().requires_grad_(True) for value in (x, weight, bias))
    grad_snapshots = _tensor_snapshot(torch, grad_values)
    output = function(*grad_values)
    torch.cuda.synchronize()
    if output.requires_grad:
        raise RuntimeError("requires-grad inputs produced a grad-tracked native output")
    _assert_inputs_unchanged(torch, grad_values, grad_snapshots)

    invalid = (
        (x.cpu(), weight, bias),
        (x.to(torch.float64), weight, bias),
        (torch.randn(3, device="cuda:0"), weight, bias),
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 2]], device="cuda:0"),
                torch.tensor([1.0, 2.0], device="cuda:0"),
                (4, 3),
            ),
            weight,
            bias,
        ),
    )
    for values in invalid:
        try:
            function(*values)
        except (TypeError, ValueError):
            continue
        raise RuntimeError("native boundary accepted an out-of-contract tensor")


def _capture_and_profile(torch: ModuleType, function: Any) -> tuple[list[str], list[str]]:
    x = torch.randn(4, 3, device="cuda:0", dtype=torch.float32)
    weight = torch.randn(3, 2, device="cuda:0", dtype=torch.float32)
    bias = torch.randn(2, device="cuda:0", dtype=torch.float32)
    stream = torch.cuda.Stream(device=0)
    if stream == torch.cuda.default_stream(device=0):
        raise RuntimeError("failed to create a non-default CUDA stream")
    stream.wait_stream(torch.cuda.current_stream(device=0))
    with torch.cuda.stream(stream):
        function(x, weight, bias)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = function(x, weight, bias)
    new_x = torch.randn_like(x)
    new_weight = torch.randn_like(weight)
    new_bias = torch.randn_like(bias)
    stream.wait_stream(torch.cuda.current_stream(device=0))
    with torch.cuda.stream(stream):
        x.copy_(new_x)
        weight.copy_(new_weight)
        bias.copy_(new_bias)
        graph.replay()
    stream.synchronize()
    eager = torch.relu(new_x @ new_weight + new_bias).mean(dim=1)
    if not torch.allclose(captured, eager, rtol=RTOL, atol=ATOL, equal_nan=True):
        raise RuntimeError("CUDA Graph replay differs from eager")

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        function(x, weight, bias)
        torch.cuda.synchronize()
    averages = list(profile.key_averages())
    names = [event.key for event in averages]
    forbidden = forbidden_profiler_events(names)
    if forbidden:
        raise RuntimeError(f"warmed native call contains transfer/copy events: {forbidden}")
    required = {
        "matmul": ("aten::matmul", "aten::mm"),
        "add": ("aten::add",),
        "relu": ("aten::relu", "aten::clamp_min"),
        "mean": ("aten::mean",),
    }
    observed: list[str] = []
    for label, candidates in required.items():
        matching = [event for event in averages if event.key in candidates]
        if not matching:
            raise RuntimeError(f"profiler omitted expected ATen operation: {label}")
        has_cuda_activity = any(
            float(
                getattr(
                    event,
                    "self_device_time_total",
                    getattr(event, "self_cuda_time_total", 0.0),
                )
            )
            > 0.0
            for event in matching
        )
        if not has_cuda_activity:
            raise RuntimeError(f"expected ATen operation had no isolated CUDA activity: {label}")
        observed.append(label)
    return forbidden, sorted(observed)


def _build_id(path: Path) -> str | None:
    output = _run(["readelf", "-n", str(path)], timeout=60)
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", output)
    return match.group(1).lower() if match else None


def _parse_ldd_checked(raw: str, label: str) -> dict[str, Path]:
    missing = re.findall(
        r"^\s*((?:libtorch|libc10)\S*)\s+=>\s+not found\s*$",
        raw,
        flags=re.MULTILINE,
    )
    if missing:
        raise EvidenceError(
            f"relevant ldd dependency was not found for {label}: {sorted(missing)}"
        )
    return parse_ldd(raw)


def _ldd_environment(torch_lib: Path) -> dict[str, str]:
    """Prepend the exact active wheel library directory for ldd resolution."""
    env = dict(os.environ)
    ambient = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        os.pathsep.join((str(torch_lib), ambient)) if ambient else str(torch_lib)
    )
    return env


def _runtime_images(torch: ModuleType, extension: Path) -> list[dict[str, object]]:
    torch_c = Path(torch._C.__file__).resolve(strict=True)
    torch_file = torch.__file__
    if torch_file is None:
        raise RuntimeError("torch package has no filesystem identity")
    torch_lib = Path(torch_file).resolve(strict=True).parent / "lib"
    loaded = parse_proc_maps(Path("/proc/self/maps").read_text(encoding="utf-8"), torch_lib)
    ldd_env = _ldd_environment(torch_lib)
    ldd_by_binary: dict[str, dict[str, Path]] = {}
    for label, binary in (("torch_c", torch_c), ("extension", extension)):
        raw = _run(["ldd", str(binary)], env=ldd_env, timeout=60)
        ldd_by_binary[label] = _parse_ldd_checked(raw, label)
    torch_rows = ldd_by_binary["torch_c"]
    extension_rows = ldd_by_binary["extension"]
    shared = sorted(
        basename
        for basename in set(torch_rows) & set(extension_rows)
        if basename.startswith(("libtorch", "libc10"))
    )
    if not shared:
        raise EvidenceError("torch._C and the generated extension share no framework image")
    for basename in shared:
        left = torch_rows[basename].resolve(strict=True)
        right = extension_rows[basename].resolve(strict=True)
        if left != right or not os.path.samefile(left, right):
            raise EvidenceError(f"ldd disagrees on framework image identity: {basename}")
    result: list[dict[str, object]] = []
    for basename, path in loaded.items():
        relative = redact_path(path, torch_lib)
        linked_rows = [
            rows[basename]
            for rows in (torch_rows, extension_rows)
            if basename in rows
        ]
        unlinked_bootstrap = (
            not linked_rows
            and basename == TORCH_GLOBAL_DEPS_BASENAME
            and relative == TORCH_GLOBAL_DEPS_BASENAME
        )
        if (not linked_rows and not unlinked_bootstrap) or any(
            linked.resolve(strict=True) != path or not os.path.samefile(linked, path)
            for linked in linked_rows
        ):
            raise EvidenceError(f"ldd does not resolve loaded image identically: {basename}")
        result.append(
            {
                "basename": basename,
                "wheel_relative_path": relative,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "build_id": _build_id(path),
                "ldd_match": bool(linked_rows),
                "shared_by_torch_c_and_extension": basename in shared,
            }
        )
    return result


def _sanitize_direct_url_identity(
    value: object, name: str = "distribution"
) -> tuple[str, str]:
    """Hash only bounded non-URL identity fields from direct_url.json."""
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid direct_url.json for {name}")
    identity: dict[str, object]
    if "vcs_info" in value:
        kind = "vcs"
        vcs = value["vcs_info"]
        if not isinstance(vcs, dict):
            raise RuntimeError(f"invalid VCS direct_url identity for {name}")
        vcs_type = vcs.get("vcs")
        commit_id = vcs.get("commit_id")
        if (
            not isinstance(vcs_type, str)
            or re.fullmatch(r"[a-z0-9_-]{1,24}", vcs_type) is None
            or not isinstance(commit_id, str)
            or re.fullmatch(r"[0-9a-fA-F]{7,64}", commit_id) is None
        ):
            raise RuntimeError(f"unsafe VCS direct_url identity for {name}")
        identity = {"kind": kind, "vcs": vcs_type, "commit_id": commit_id.lower()}
    elif value.get("dir_info", {}).get("editable") is True:
        kind = "editable"
        identity = {"kind": kind, "editable": True}
    elif "archive_info" in value:
        kind = "archive"
        archive = value["archive_info"]
        if not isinstance(archive, dict):
            raise RuntimeError(f"invalid archive direct_url identity for {name}")
        hashes = archive.get("hashes", {})
        if not isinstance(hashes, dict) or any(
            not isinstance(algorithm, str)
            or re.fullmatch(r"[a-z0-9_-]{1,24}", algorithm) is None
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{32,128}", digest) is None
            for algorithm, digest in hashes.items()
        ):
            raise RuntimeError(f"unsafe archive direct_url identity for {name}")
        identity = {
            "kind": kind,
            "hashes": {key: value.lower() for key, value in sorted(hashes.items())},
        }
    else:
        kind = "other"
        identity = {"kind": kind}
    # Never hash the raw document: it may contain URL credentials or paths.
    return kind, _hash_json(identity)


def _direct_url_identity(name: str) -> tuple[str | None, str | None]:
    distribution = importlib.metadata.distribution(name)
    text = distribution.read_text("direct_url.json")
    if text is None:
        return None, None
    return _sanitize_direct_url_identity(json.loads(text), name)


def _packages() -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("torch", "rextio", "rextio-device-cuda", "rextio-torch"):
        distribution = importlib.metadata.distribution(name)
        kind, digest = _direct_url_identity(name)
        result[name] = {
            "version": distribution.version,
            "direct_url_kind": kind,
            "direct_url_sha256": digest,
        }
    return result


def _hash_json(value: object) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value)).hexdigest()


def _write_atomic(path: Path, document: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json(document))
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--provider-root", type=Path)
    parser.add_argument("--expected-torch-commit", required=True)
    parser.add_argument("--sm", required=True, choices=sorted(ALLOWED_SMS))
    return parser.parse_args()


def main() -> int:
    """Execute the manual candidate and atomically emit non-promoting evidence."""
    args = _parse_args()
    torch_root = args.torch_root.resolve(strict=True)
    workspace = torch_root.parent
    core_root = (args.core_root or workspace / "rextio").resolve(strict=True)
    provider_root = (args.provider_root or workspace / "rextio-device-cuda").resolve(strict=True)
    if (
        sys.version_info[:2] != (3, 11)
        or platform.python_implementation() != "CPython"
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
    ):
        raise SystemExit("candidate requires CPython 3.11 on Linux x86_64")
    if os.environ.get("LIBTORCH_USE_PYTORCH") != "1":
        raise SystemExit("LIBTORCH_USE_PYTORCH=1 is required")
    if "LIBTORCH_BYPASS_VERSION_CHECK" in os.environ:
        raise SystemExit("LIBTORCH_BYPASS_VERSION_CHECK is forbidden")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_torch_commit):
        raise SystemExit("--expected-torch-commit must be a full lowercase commit")

    _require_checkout(torch_root, args.expected_torch_commit, ancestor=TORCH_BASE)
    _require_checkout(core_root, CORE_COMMIT)
    _require_checkout(provider_root, PROVIDER_COMMIT)
    probe = _build_probe(provider_root)
    probe_sha256_before = sha256_file(probe)

    import torch
    import rextio
    import rextio_device_cuda
    import rextio_torch

    if torch.__version__.split("+")[0] != "2.11.0" or not torch.version.cuda:
        raise RuntimeError("exact CUDA-enabled torch 2.11.0 wheel is required")
    for module, root, label in (
        (rextio, core_root, "rextio"),
        (rextio_device_cuda, provider_root, "rextio-device-cuda"),
        (rextio_torch, torch_root, "rextio-torch"),
    ):
        _require_module_under_root(module, root, label)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("real CUDA device 0 is unavailable")
    properties = torch.cuda.get_device_properties(0)
    observed_sm = f"sm_{properties.major}{properties.minor}"
    if observed_sm != args.sm:
        raise RuntimeError(f"selected SM {args.sm} differs from device {observed_sm}")

    work = args.work_dir.resolve()
    if work.exists():
        raise RuntimeError("--work-dir must not already exist")
    if args.output.exists():
        raise RuntimeError("--output must not already exist")
    work.mkdir(parents=True)
    result = _generate(work, probe, args.sm)
    probe_sha256_after = sha256_file(probe)
    if probe_sha256_after != probe_sha256_before:
        raise RuntimeError("provider probe binary changed during preflight")
    rust = (result.layout.rust_dir / "src/lib.rs").read_text(encoding="utf-8")
    forbidden_static = (".to_device(", ".cpu(", ".cuda(", "to_kind(")
    if any(token in rust for token in forbidden_static):
        raise RuntimeError("generated Rust contains a transfer/conversion token")
    cargo_environment = _bound_cargo_environment(torch)
    extension = _build_extension(result.layout.rust_dir, cargo_environment)
    module = _load_extension(extension)  # torch was intentionally imported first
    function = getattr(module, "cuda_app__kernels__inference")

    errors = [_call_case(torch, function, contiguous=True), _call_case(torch, function, contiguous=False)]
    _requires_grad_and_boundary_checks(torch, function)
    profiler_events, expected_cuda_ops = _capture_and_profile(torch, function)
    images = _runtime_images(torch, extension)
    [provider_plan] = result.device_provider_plans
    report = provider_plan["report"]
    authorization = provider_plan["lowering_authorization"]
    lock = provider_plan["lock"]
    probe_observations = report["observations"]
    observation_map = {row["key"]: row["value"] for row in probe_observations}
    driver_version = int(observation_map["driver.version"])
    lock_profile_sha256 = lock["artifact_profile_sha256"]
    if authorization["artifact_profile_sha256"] != lock_profile_sha256:
        raise RuntimeError("authorization and provider lock artifact profiles differ")
    max_scaled_error = max(item[2] for item in errors)
    if not math.isfinite(max_scaled_error) or max_scaled_error > 1.0:
        raise RuntimeError("native numerical error exceeds the frozen allclose scale")
    rust_dir = result.layout.rust_dir

    payload = {
        "support_claim": False,
        "certification_ready": False,
        "kernel_executed": True,
        "scope": {
            "python": "3.11",
            "platform": "linux-x86_64",
            "torch": "2.11.0",
            "tch": "0.24.0",
            "dtype": "float32",
            "ranks": [1, 2],
            "device": "cuda:0",
            "operations": ["matmul", "bias-add", "relu", "mean-dim1"],
        },
        "source": {
            "torch_commit": args.expected_torch_commit,
            "torch_base_ancestor": TORCH_BASE,
            "core_commit": CORE_COMMIT,
            "provider_commit": PROVIDER_COMMIT,
        },
        "packages": _packages(),
        "toolchain": {
            "rustc": _run(["rustc", "+1.93.1", "--version"], timeout=60),
            "cargo": _run(["cargo", "+1.93.1", "--version"], timeout=60),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "device": {
            "ordinal": 0,
            "name": properties.name,
            "sm": observed_sm,
            "driver_version": driver_version,
            "torch_cuda": torch.version.cuda,
            "cuda_runtime": torch._C._cuda_getCompiledVersion(),
        },
        "orchestration": {
            "provider_id": CORE_PROVIDER_ID,
            "capability_id": CORE_CAPABILITY_ID,
            "support_claim": False,
            "artifact_profile_sha256": authorization["artifact_profile_sha256"],
            "provider_lock_artifact_profile_sha256": lock_profile_sha256,
            "authorization_sha256": _hash_json(authorization),
            "provider_lock_sha256": _hash_json(lock),
            "probe_sha256": probe_sha256_before,
            "probe_observations_sha256": _hash_json(probe_observations),
        },
        "artifact": {
            "ordered_rule_ids": list(E2_RULES),
            "generated_lib_rs_sha256": sha256_file(rust_dir / "src/lib.rs"),
            "generated_cargo_toml_sha256": sha256_file(rust_dir / "Cargo.toml"),
            "generated_cargo_lock_sha256": sha256_file(rust_dir / "Cargo.lock"),
            "extension_sha256": sha256_file(extension),
        },
        "execution": {
            "cases": ["contiguous", "noncontiguous"],
            "rtol": RTOL,
            "atol": ATOL,
            "max_abs_error": max(item[0] for item in errors),
            "max_rel_error": max(item[1] for item in errors),
            "max_scaled_error": max_scaled_error,
            "numerically_equivalent": True,
            "output_contract": {
                "device": "cuda:0",
                "dtype": "float32",
                "rank": 1,
                "layout": "torch.strided",
                "requires_grad": False,
            },
            "inputs_unmutated": True,
            "output_lifetime": True,
            "cuda_graph_capture": True,
            "cuda_graph_replay": True,
            "non_default_stream": True,
            "static_transfer_scan": True,
            "profiler_transfer_events": profiler_events,
            "expected_cuda_aten_ops": expected_cuda_ops,
            "cuda_kernel_activity": True,
        },
        "runtime_images": images,
    }
    document = make_envelope(payload)
    verify_document(document)
    _write_atomic(args.output, document)
    print(json.dumps({"evidence": args.output.name, "support_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
