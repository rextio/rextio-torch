"""Offline, fail-closed verifier for the bounded Torch CUDA E2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "rextio-torch.cuda-e2.real-nvidia.v1"
MAX_EVIDENCE_BYTES = 1_048_576
MAX_DEPTH = 12
TORCH_GLOBAL_DEPS_BASENAME = "libtorch_global_deps.so"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_SMS = frozenset(
    {"sm_60", "sm_61", "sm_70", "sm_72", "sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"}
)
TOP_KEYS = frozenset(
    {
        "support_claim",
        "certification_ready",
        "kernel_executed",
        "scope",
        "source",
        "packages",
        "toolchain",
        "device",
        "orchestration",
        "artifact",
        "execution",
        "runtime_images",
    }
)
ENVELOPE_KEYS = frozenset({"schema", "payload", "payload_sha256"})


class EvidenceError(ValueError):
    """Raised when evidence is malformed, tampered with, or overclaiming."""


def canonical_json(document: object) -> bytes:
    """Return the sole accepted deterministic JSON encoding."""
    try:
        text = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise EvidenceError("evidence is not finite canonical JSON") from None
    return (text + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    """Hash a regular file without following a caller-provided identity claim."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact_path(path: Path, root: Path) -> str:
    """Return a root-relative POSIX path and reject escaping identities."""
    resolved = path.resolve(strict=True)
    base = root.resolve(strict=True)
    try:
        relative = resolved.relative_to(base)
    except ValueError:
        raise EvidenceError("path is outside the declared root") from None
    text = relative.as_posix()
    if not text or text.startswith("/") or ".." in PurePosixPath(text).parts:
        raise EvidenceError("invalid redacted path")
    return text


def parse_proc_maps(text: str, torch_lib: Path) -> dict[str, Path]:
    """Resolve one canonical torch-wheel image for every relevant loaded basename."""
    root = torch_lib.resolve(strict=True)
    rows: dict[str, set[Path]] = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        candidate = Path(fields[5])
        name = candidate.name
        if not (name.startswith("libtorch") or name.startswith("libc10")):
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            raise EvidenceError(f"relevant runtime image is outside torch/lib: {name}") from None
        rows.setdefault(name, set()).add(resolved)
    if not rows:
        raise EvidenceError("no loaded libtorch/c10 image found")
    duplicated = {name for name, paths in rows.items() if len(paths) != 1}
    if duplicated:
        raise EvidenceError(f"duplicate runtime image basename: {sorted(duplicated)}")
    return {name: next(iter(paths)) for name, paths in sorted(rows.items())}


def parse_ldd(text: str) -> dict[str, Path]:
    """Parse only resolved absolute ldd dependency rows."""
    result: dict[str, Path] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(\S+)\s+=>\s+(/\S+)\s+\(0x[0-9a-fA-F]+\)\s*$", line)
        if match:
            result[match.group(1)] = Path(match.group(2))
    return result


def forbidden_profiler_events(names: list[str]) -> list[str]:
    """Return transfer/copy events forbidden in the warmed native call."""
    patterns = ("aten::to", "aten::_to_copy", "memcpy htod", "memcpy dtoh")
    return sorted(name for name in names if any(item in name.lower() for item in patterns))


def _mapping(value: object, name: str, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise EvidenceError(f"{name} has an unknown or missing field")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise EvidenceError(f"{name} is not sha256")
    return value


def _bounded_depth(value: object, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise EvidenceError("evidence nesting exceeds the bound")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise EvidenceError("evidence keys must be strings")
        for item in value.values():
            _bounded_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _bounded_depth(item, depth + 1)


def _reject_path_or_url_leaks(value: object, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, item in value.items():
            _reject_path_or_url_leaks(item, child_key)
    elif isinstance(value, list):
        for item in value:
            _reject_path_or_url_leaks(item, key)
    elif isinstance(value, str) and key != "wheel_relative_path":
        if value.startswith(("/", "file:")) or "://" in value:
            raise EvidenceError("evidence contains an absolute path or URL")


def make_envelope(payload: object) -> dict[str, object]:
    """Bind the canonical payload without a circular whole-file self-hash."""
    if not isinstance(payload, dict):
        raise EvidenceError("payload must be an object")
    return {
        "schema": SCHEMA,
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
    }


def verify_document(document: object) -> None:
    """Verify exact envelope/schema/invariants without ever promoting support."""
    envelope = _mapping(document, "envelope", ENVELOPE_KEYS)
    _bounded_depth(envelope)
    _reject_path_or_url_leaks(envelope)
    if envelope["schema"] != SCHEMA:
        raise EvidenceError("unknown evidence schema")
    digest = _sha(envelope["payload_sha256"], "payload_sha256")
    if digest != hashlib.sha256(canonical_json(envelope["payload"])).hexdigest():
        raise EvidenceError("payload hash mismatch")
    root = _mapping(envelope["payload"], "payload", TOP_KEYS)
    if (
        root["support_claim"] is not False
        or root["certification_ready"] is not False
        or root["kernel_executed"] is not True
    ):
        raise EvidenceError("evidence overclaims or did not execute a kernel")

    scope = _mapping(
        root["scope"],
        "scope",
        frozenset({"python", "platform", "torch", "tch", "dtype", "ranks", "device", "operations"}),
    )
    expected_scope = {
        "python": "3.11",
        "platform": "linux-x86_64",
        "torch": "2.11.0",
        "tch": "0.24.0",
        "dtype": "float32",
        "ranks": [1, 2],
        "device": "cuda:0",
        "operations": ["matmul", "bias-add", "relu", "mean-dim1"],
    }
    if scope != expected_scope:
        raise EvidenceError("scope differs from the frozen E2 slice")

    source = _mapping(
        root["source"],
        "source",
        frozenset({"torch_commit", "torch_base_ancestor", "core_commit", "provider_commit"}),
    )
    for key, value in source.items():
        if COMMIT_RE.fullmatch(value) is None:
            raise EvidenceError(f"{key} is not an exact commit")
    if source["torch_base_ancestor"] != "7d6fb1b606bfab6530a7ff96317081b2fd1c1b22":
        raise EvidenceError("wrong Torch candidate ancestry")
    if source["core_commit"] != "7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0":
        raise EvidenceError("wrong Core commit")
    if source["provider_commit"] != "a5fb427e91710b65f54ee5b8e33706c45840cf9c":
        raise EvidenceError("wrong provider commit")

    packages = root["packages"]
    if not isinstance(packages, dict) or set(packages) != {
        "torch",
        "rextio",
        "rextio-device-cuda",
        "rextio-torch",
    }:
        raise EvidenceError("package identity set differs from the frozen environment")
    for name, row_value in packages.items():
        row = _mapping(
            row_value,
            f"packages.{name}",
            frozenset({"version", "direct_url_kind", "direct_url_sha256"}),
        )
        if not isinstance(row["version"], str) or not row["version"]:
            raise EvidenceError(f"missing package version: {name}")
        if row["direct_url_kind"] not in {None, "archive", "editable", "vcs", "other"}:
            raise EvidenceError(f"invalid direct_url kind: {name}")
        if row["direct_url_sha256"] is not None:
            _sha(row["direct_url_sha256"], f"{name} direct_url identity")
    if packages["torch"]["version"].split("+")[0] != "2.11.0":
        raise EvidenceError("wrong torch distribution")

    toolchain = _mapping(
        root["toolchain"],
        "toolchain",
        frozenset({"rustc", "cargo", "python_implementation", "python_version"}),
    )
    if (
        not isinstance(toolchain["rustc"], str)
        or not toolchain["rustc"].startswith("rustc 1.93.1 ")
        or not isinstance(toolchain["cargo"], str)
        or not toolchain["cargo"].startswith("cargo 1.93.1 ")
    ):
        raise EvidenceError("wrong Rust toolchain")
    if toolchain["python_implementation"] != "CPython" or not str(
        toolchain["python_version"]
    ).startswith("3.11."):
        raise EvidenceError("wrong Python runtime")

    device = _mapping(
        root["device"],
        "device",
        frozenset({"ordinal", "name", "sm", "driver_version", "torch_cuda", "cuda_runtime"}),
    )
    if device["ordinal"] != 0 or device["sm"] not in ALLOWED_SMS:
        raise EvidenceError("device selection is outside the frozen scope")
    if type(device["driver_version"]) is not int or device["driver_version"] < 12000:
        raise EvidenceError("driver version is below the provider floor")
    if (
        not isinstance(device["name"], str)
        or not 1 <= len(device["name"]) <= 160
        or not isinstance(device["torch_cuda"], str)
        or not device["torch_cuda"]
        or type(device["cuda_runtime"]) is not int
        or device["cuda_runtime"] <= 0
    ):
        raise EvidenceError("invalid bounded device/runtime identity")

    orchestration = _mapping(
        root["orchestration"],
        "orchestration",
        frozenset(
            {
                "provider_id",
                "capability_id",
                "support_claim",
                "artifact_profile_sha256",
                "provider_lock_artifact_profile_sha256",
                "authorization_sha256",
                "provider_lock_sha256",
                "probe_sha256",
                "probe_observations_sha256",
            }
        ),
    )
    if (
        orchestration["provider_id"] != "rextio-device-cuda"
        or orchestration["capability_id"] != "cuda-libtorch-linux-x86_64"
        or orchestration["support_claim"] is not False
    ):
        raise EvidenceError("wrong or overclaiming provider authorization")
    for key in (
        "artifact_profile_sha256",
        "provider_lock_artifact_profile_sha256",
        "authorization_sha256",
        "provider_lock_sha256",
        "probe_sha256",
        "probe_observations_sha256",
    ):
        _sha(orchestration[key], key)
    if (
        orchestration["artifact_profile_sha256"]
        != orchestration["provider_lock_artifact_profile_sha256"]
    ):
        raise EvidenceError("authorization/provider-lock artifact profiles differ")

    artifact = _mapping(
        root["artifact"],
        "artifact",
        frozenset(
            {
                "ordered_rule_ids",
                "generated_lib_rs_sha256",
                "generated_cargo_toml_sha256",
                "generated_cargo_lock_sha256",
                "extension_sha256",
            }
        ),
    )
    if artifact["ordered_rule_ids"] != [
        "rextio-torch/cuda0-matmul-f32-2d",
        "rextio-torch/cuda0-bias-add-f32-2d-1d",
        "rextio-torch/cuda0-relu-f32-2d",
        "rextio-torch/cuda0-mean-dim1-f32-2d",
    ]:
        raise EvidenceError("artifact does not bind the exact ordered E2 rules")
    for key in (
        "generated_lib_rs_sha256",
        "generated_cargo_toml_sha256",
        "generated_cargo_lock_sha256",
        "extension_sha256",
    ):
        _sha(artifact[key], key)

    execution = _mapping(
        root["execution"],
        "execution",
        frozenset(
            {
                "cases",
                "rtol",
                "atol",
                "max_abs_error",
                "max_rel_error",
                "max_scaled_error",
                "numerically_equivalent",
                "output_contract",
                "inputs_unmutated",
                "output_lifetime",
                "cuda_graph_capture",
                "cuda_graph_replay",
                "non_default_stream",
                "static_transfer_scan",
                "profiler_transfer_events",
                "expected_cuda_aten_ops",
                "cuda_kernel_activity",
            }
        ),
    )
    if execution["cases"] != ["contiguous", "noncontiguous"]:
        raise EvidenceError("both layout cases were not executed")
    exact_ops = ["add", "matmul", "mean", "relu"]
    derived_kernel_executed = (
        execution["cuda_kernel_activity"] is True
        and execution["expected_cuda_aten_ops"] == exact_ops
    )
    if root["kernel_executed"] is not derived_kernel_executed:
        raise EvidenceError("kernel_executed is not derived from exact CUDA activity")
    if (
        execution["output_contract"]
        != {
            "device": "cuda:0",
            "dtype": "float32",
            "rank": 1,
            "layout": "torch.strided",
            "requires_grad": False,
        }
        or execution["inputs_unmutated"] is not True
        or execution["output_lifetime"] is not True
        or execution["cuda_graph_capture"] is not True
        or execution["cuda_graph_replay"] is not True
        or execution["non_default_stream"] is not True
        or execution["static_transfer_scan"] is not True
        or execution["profiler_transfer_events"] != []
        or execution["expected_cuda_aten_ops"] != exact_ops
        or execution["cuda_kernel_activity"] is not True
        or execution["numerically_equivalent"] is not True
    ):
        raise EvidenceError("execution invariants are incomplete")
    for key in ("rtol", "atol", "max_abs_error", "max_rel_error", "max_scaled_error"):
        value = execution[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise EvidenceError(f"invalid numerical field: {key}")
    if execution["rtol"] > 1e-5 or execution["atol"] > 1e-6:
        raise EvidenceError("numerical tolerances are weaker than frozen bounds")
    if execution["max_scaled_error"] > 1.0:
        raise EvidenceError("reported numerical error exceeds frozen allclose scale")

    images = root["runtime_images"]
    if not isinstance(images, list) or not images:
        raise EvidenceError("runtime image identity is absent")
    seen: set[str] = set()
    for index, row_value in enumerate(images):
        row = _mapping(
            row_value,
            f"runtime_images[{index}]",
            frozenset(
                {
                    "basename",
                    "wheel_relative_path",
                    "sha256",
                    "size",
                    "build_id",
                    "ldd_match",
                    "shared_by_torch_c_and_extension",
                }
            ),
        )
        basename = row["basename"]
        relative = row["wheel_relative_path"]
        if (
            not isinstance(basename, str)
            or basename in seen
            or not (basename.startswith("libtorch") or basename.startswith("libc10"))
            or not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in PurePosixPath(relative).parts
        ):
            raise EvidenceError("invalid or duplicate runtime image identity")
        seen.add(basename)
        _sha(row["sha256"], "runtime image sha256")
        if type(row["size"]) is not int or row["size"] <= 0:
            raise EvidenceError("invalid runtime image size")
        if row["build_id"] is not None and (
            not isinstance(row["build_id"], str)
            or re.fullmatch(r"[0-9a-f]+", row["build_id"]) is None
        ):
            raise EvidenceError("invalid ELF build id")
        if type(row["ldd_match"]) is not bool:
            raise EvidenceError("runtime ldd identity is not boolean")
        if type(row["shared_by_torch_c_and_extension"]) is not bool:
            raise EvidenceError("runtime shared-image identity is not boolean")
        if row["ldd_match"] is False and (
            basename != TORCH_GLOBAL_DEPS_BASENAME
            or relative != TORCH_GLOBAL_DEPS_BASENAME
            or row["shared_by_torch_c_and_extension"] is not False
        ):
            raise EvidenceError("runtime image was not bound through ldd")
    if not any(row["basename"].startswith("libtorch") for row in images):
        raise EvidenceError("no shared libtorch framework image was proven")
    if not any(row["shared_by_torch_c_and_extension"] is True for row in images):
        raise EvidenceError("torch._C and extension shared no bound framework image")


def verify_file(path: Path) -> str:
    """Verify canonical bytes and return an out-of-band digest."""
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_EVIDENCE_BYTES:
        raise EvidenceError("evidence size is outside the bound")
    try:
        document = json.loads(
            raw,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                EvidenceError("non-finite JSON constant")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EvidenceError("evidence is not valid JSON") from None
    if raw != canonical_json(document):
        raise EvidenceError("evidence is not canonically serialized")
    verify_document(document)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    """Verify one evidence file without changing its support status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    digest = verify_file(args.evidence)
    print(json.dumps({"verified": True, "support_claim": False, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
