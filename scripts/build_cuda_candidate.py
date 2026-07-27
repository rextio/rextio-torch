"""Generate and compile the CUDA E2 candidate through real Core orchestration.

The provider probe is synthetic and deterministic.  The generated cdylib is
compiled exactly once and is never loaded or executed, so this remains
build-only evidence rather than a CUDA support or real-device claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

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

CORE_PROVIDER_ID = "rextio-device-cuda"
CORE_CAPABILITY_ID = "cuda-libtorch-linux-x86_64"
CORE_TARGET = "x86_64-unknown-linux-gnu"
E2_RULES = (
    "rextio-torch/cuda0-matmul-f32-2d",
    "rextio-torch/cuda0-bias-add-f32-2d-1d",
    "rextio-torch/cuda0-relu-f32-2d",
    "rextio-torch/cuda0-mean-dim1-f32-2d",
)

KERNELS = """\
from rextio_torch.types import TensorF32Cuda0_1D, TensorF32Cuda0_2D


def inference(
    x: TensorF32Cuda0_2D,
    weight: TensorF32Cuda0_2D,
    bias: TensorF32Cuda0_1D,
) -> TensorF32Cuda0_1D:
    hidden = x @ weight
    biased = hidden + bias
    activated = biased.relu()
    return activated.mean(dim=1)
"""


@dataclass
class FixedRunner:
    """Return one reviewed synthetic report without inspecting the host."""

    report: object
    calls: int = 0

    def run(self) -> object:
        """Return the fixed report and record the single preflight call."""
        self.calls += 1
        return self.report


@dataclass(frozen=True)
class _Distribution:
    name: str = "rextio-device-cuda"
    version: str = "0.1.0"


class _DeviceEntryPoint:
    group = DEVICE_PROVIDER_ENTRY_POINT
    name = CORE_PROVIDER_ID
    value = "rextio_device_cuda.provider:provider"
    dist = _Distribution()

    def __init__(self, provider: object) -> None:
        self._provider = provider

    def load(self) -> object:
        """Return the real provider configured with a synthetic probe runner."""
        return self._provider


class _PluginEntryPoint:
    name = PLUGIN_ID

    def load(self):
        """Return the real rextio-torch entry-point factory."""
        return plugin


def _probe_report() -> object:
    from rextio_device_cuda.probe import (
        CudaDeviceRecord,
        CudaProbeReport,
        ProbeTarget,
    )

    return CudaProbeReport(
        target=ProbeTarget(os="linux", arch="x86_64", environment="gnu"),
        platform_supported=True,
        status="probe-complete",
        reason_code=None,
        driver_loaded=True,
        driver_version=12_080,
        device_count=1,
        cuda_result=0,
        devices=(
            CudaDeviceRecord(
                ordinal=0,
                name="Synthetic NVIDIA Build-Only GPU",
                compute_major=8,
                compute_minor=0,
                sm="sm_80",
            ),
        ),
    )


def _write_fixture(root: Path) -> None:
    package = root / "src" / "cuda_app"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNELS, encoding="utf-8")


def _assert_orchestration(result: object, runner: FixedRunner) -> Path:
    if runner.calls != 1:
        raise RuntimeError(f"expected one synthetic provider probe, got {runner.calls}")

    plan = result.plan
    [profile] = plan.artifact_profiles
    if profile.target_triple != CORE_TARGET:
        raise RuntimeError(f"unexpected target triple: {profile.target_triple}")
    [device] = profile.device_requirements
    if device.to_dict() != {
        "logical_device": "gpu:0",
        "backend": "cuda",
        "runtime": "libtorch",
        "features": ["inference", "no-grad"],
        "layouts": ["strided"],
        "memory_spaces": ["device"],
        "architectures": ["sm_80"],
        "reuse_domain_runtime": True,
    }:
        raise RuntimeError(f"unexpected derived device requirement: {device.to_dict()}")
    runtime_rows = {
        (item.name, item.version, item.features)
        for item in profile.runtime_requirements
    }
    required_rows = {
        ("libtorch", "2.11.0", ("cuda", "pytorch-wheel")),
        ("tch", "0.24.0", ("cuda",)),
    }
    if not required_rows.issubset(runtime_rows):
        raise RuntimeError(f"missing runtime requirements: {runtime_rows}")

    [provider_plan] = result.device_provider_plans
    if provider_plan["manifest"]["provider_id"] != CORE_PROVIDER_ID:
        raise RuntimeError("wrong resolved device provider")
    capability_ids = {
        item["id"] for item in provider_plan["manifest"]["capabilities"]
    }
    if CORE_CAPABILITY_ID not in capability_ids:
        raise RuntimeError("libtorch reuse capability absent from resolved manifest")
    report = provider_plan["report"]
    if report["support_claim"] is not False or report["certification_tier"] != "build-only":
        raise RuntimeError(f"provider report overclaims support: {report}")
    authorization = provider_plan["lowering_authorization"]
    if (
        authorization["provider_id"] != CORE_PROVIDER_ID
        or authorization["capability_id"] != CORE_CAPABILITY_ID
        or authorization["logical_device"] != "gpu:0"
        or authorization["backend"] != "cuda"
        or authorization["runtime"] != "libtorch"
        or authorization["reuse_domain_runtime"] is not True
        or authorization["features"] != ["inference", "no-grad"]
        or authorization["layouts"] != ["strided"]
        or authorization["memory_spaces"] != ["device"]
    ):
        raise RuntimeError(f"unexpected lowering authorization: {authorization}")
    if not re.fullmatch(r"[0-9a-f]{64}", authorization["artifact_profile_sha256"]):
        raise RuntimeError("authorization omitted the artifact profile hash")
    if (
        authorization["artifact_profile_sha256"]
        != provider_plan["lock"]["artifact_profile_sha256"]
    ):
        raise RuntimeError("authorization/profile lock hashes differ")

    contribution = provider_plan["contribution"]
    for empty_key in (
        "cargo_features",
        "native_libraries",
        "package_references",
        "generated_helper_ids",
        "runtime_check_ids",
    ):
        if contribution[empty_key] != []:
            raise RuntimeError(f"libtorch reuse contribution owns {empty_key}")
    resources = contribution["resource_contracts"]
    if {item["resource_kind"] for item in resources} != {
        "framework.allocator",
        "framework.current-stream",
        "framework.tensor",
    }:
        raise RuntimeError(f"unexpected framework borrow contracts: {resources}")
    if any(
        item["owner"] != "framework"
        or item["access"] != "borrow-validate"
        or item["may_allocate"]
        or item["may_replace"]
        or item["may_synchronize"]
        for item in resources
    ):
        raise RuntimeError(f"provider contribution is not borrow-only: {resources}")

    rust_dir = result.layout.rust_dir
    cargo = tomllib.loads((rust_dir / "Cargo.toml").read_text(encoding="utf-8"))
    tch = cargo["dependencies"]["tch"]
    if tch != {"version": "=0.24.0", "features": ["python-extension"]}:
        raise RuntimeError(f"unexpected generated tch dependency: {tch}")

    rust = (rust_dir / "src" / "lib.rs").read_text(encoding="utf-8")
    for token in (
        "tch::Device::Cuda(0)",
        "__rxttorch_extract_f32_cuda0_2d",
        "__rxttorch_extract_f32_cuda0_1d",
        "__rxttorch_materialize_f32_cuda0_2d",
        "__rxttorch_materialize_f32_cuda0_1d",
        "__rxttorch_require_python_strided",
        "__rxttorch_require_native_strided",
        'getattr("layout")',
        '"torch.strided"',
        "tensor.is_sparse()",
        "tensor.is_mkldnn()",
        "__rxttorch_matmul_function_scoped",
        "__rxttorch_add_function_scoped",
        "__rxttorch_relu_function_scoped",
        "__rxttorch_mean_dim1_keepdim_false_function_scoped",
    ):
        if token not in rust:
            raise RuntimeError(f"generated Rust omitted {token}")
    if any(token in rust for token in (".to_device(", ".cpu(", ".cuda(")):
        raise RuntimeError("generated Rust contains a forbidden device transfer")

    function_start = rust.find("fn cuda_app__kernels__inference")
    if function_start < 0:
        raise RuntimeError("generated Rust omitted the inference function")
    function_body = rust[function_start : function_start + 4_000]
    calls = (
        "__rxttorch_matmul_function_scoped(",
        "__rxttorch_add_function_scoped(",
        "__rxttorch_relu_function_scoped(",
        "__rxttorch_mean_dim1_keepdim_false_function_scoped(",
    )
    positions = tuple(function_body.find(call) for call in calls)
    if any(position < 0 for position in positions) or positions != tuple(sorted(positions)):
        raise RuntimeError(f"generated CUDA call order changed: {positions}")
    return rust_dir


def generate_candidate(root: Path) -> Path:
    """Run analyzer, real provider preflight, authorization, and Core codegen."""
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
        raise RuntimeError("analyzer did not accept the exact four-op CUDA E2 chain")

    runner = FixedRunner(_probe_report())
    provider = CudaDeviceProvider(
        CudaProviderConfig(),
        probe_runner=runner,
    )
    result = generate_source_artifact(
        root,
        analysis,
        "cpython",
        target_plan=TargetPlan(TargetSpec(), registry),
        device_selection=DeviceProviderSelection(
            CORE_PROVIDER_ID,
            CORE_CAPABILITY_ID,
        ),
        device_options=DeviceProviderOptions(
            values=(("device_ordinal", "0"), ("sm", "sm_80"))
        ),
        device_entry_points=(_DeviceEntryPoint(provider),),
    )
    if result.native_source.status != "generated":
        raise RuntimeError(f"Core source generation failed: {result.native_source}")
    return _assert_orchestration(result, runner)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_cuda_enabled_torch(torch_module: object) -> str:
    """Return the pinned libtorch CUDA version or fail closed."""
    version_namespace = getattr(torch_module, "version", None)
    cuda_version = getattr(version_namespace, "cuda", None)
    if not isinstance(cuda_version, str) or not cuda_version.strip():
        raise SystemExit(
            "CUDA build-only candidate requires a CUDA-enabled PyTorch/libtorch "
            "(torch.version.cuda must be non-empty)"
        )
    return cuda_version


def main() -> int:
    """Generate once, build once, and never load or execute the artifact."""
    args = _parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("CUDA build-only candidate requires CPython 3.11")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise SystemExit("CUDA build-only candidate requires Linux x86_64")
    if os.environ.get("LIBTORCH_USE_PYTORCH") != "1":
        raise SystemExit("LIBTORCH_USE_PYTORCH=1 is required")
    if "LIBTORCH_BYPASS_VERSION_CHECK" in os.environ:
        raise SystemExit("LIBTORCH_BYPASS_VERSION_CHECK is forbidden")

    import torch

    if torch.__version__.split("+")[0] != "2.11.0":
        raise SystemExit(f"expected torch 2.11.0, got {torch.__version__}")
    _require_cuda_enabled_torch(torch)

    root = args.output.resolve()
    rust_dir = generate_candidate(root)
    completed = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(rust_dir / "Cargo.toml")],
        check=False,
        env=os.environ,
    )
    if completed.returncode == 0:
        release_dir = rust_dir / "target" / "release"
        linked = tuple(release_dir.glob("*_rextio_native*.so"))
        if len(linked) != 1 or not linked[0].is_file() or linked[0].stat().st_size == 0:
            raise RuntimeError(
                f"cargo succeeded without one linked Linux cdylib: {linked}"
            )
        summary = {
            "support_claim": False,
            "synthetic_probe": True,
            "cuda_function_executed": False,
            "cargo_builds": 1,
            "linked_cdylib": linked[0].name,
        }
        print(json.dumps(summary, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
