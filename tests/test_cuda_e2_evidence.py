"""GPU-free tests for the real-NVIDIA candidate evidence boundary."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import scripts.certify_cuda_candidate as certifier
from scripts.verify_cuda_e2_evidence import (
    EvidenceError,
    canonical_json,
    forbidden_profiler_events,
    make_envelope,
    parse_ldd,
    parse_proc_maps,
    redact_path,
    verify_document,
    verify_file,
)
from scripts.certify_cuda_candidate import (
    _discover_extension,
    _parse_ldd_checked,
    _require_module_under_root,
    _sanitize_direct_url_identity,
)


def _payload() -> dict[str, object]:
    return {
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
            "torch_commit": "1" * 40,
            "torch_base_ancestor": "7d6fb1b606bfab6530a7ff96317081b2fd1c1b22",
            "core_commit": "7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0",
            "provider_commit": "a5fb427e91710b65f54ee5b8e33706c45840cf9c",
        },
        "packages": {
            name: {
                "version": "2.11.0" if name == "torch" else "0.1.0",
                "direct_url_kind": None,
                "direct_url_sha256": None,
            }
            for name in ("torch", "rextio", "rextio-device-cuda", "rextio-torch")
        },
        "toolchain": {
            "rustc": "rustc 1.93.1 (fake 2026-01-01)",
            "cargo": "cargo 1.93.1 (fake 2026-01-01)",
            "python_implementation": "CPython",
            "python_version": "3.11.14",
        },
        "device": {
            "ordinal": 0,
            "name": "Fixture NVIDIA GPU",
            "sm": "sm_80",
            "driver_version": 12080,
            "torch_cuda": "12.8",
            "cuda_runtime": 12080,
        },
        "orchestration": {
            "provider_id": "rextio-device-cuda",
            "capability_id": "cuda-libtorch-linux-x86_64",
            "support_claim": False,
            "artifact_profile_sha256": "2" * 64,
            "provider_lock_artifact_profile_sha256": "2" * 64,
            "authorization_sha256": "3" * 64,
            "provider_lock_sha256": "4" * 64,
            "probe_sha256": "5" * 64,
            "probe_observations_sha256": "6" * 64,
        },
        "artifact": {
            "ordered_rule_ids": [
                "rextio-torch/cuda0-matmul-f32-2d",
                "rextio-torch/cuda0-bias-add-f32-2d-1d",
                "rextio-torch/cuda0-relu-f32-2d",
                "rextio-torch/cuda0-mean-dim1-f32-2d",
            ],
            "generated_lib_rs_sha256": "8" * 64,
            "generated_cargo_toml_sha256": "9" * 64,
            "generated_cargo_lock_sha256": "a" * 64,
            "extension_sha256": "b" * 64,
        },
        "execution": {
            "cases": ["contiguous", "noncontiguous"],
            "rtol": 1e-5,
            "atol": 1e-6,
            "max_abs_error": 1e-7,
            "max_rel_error": 1e-7,
            "max_scaled_error": 0.5,
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
            "profiler_transfer_events": [],
            "expected_cuda_aten_ops": ["add", "matmul", "mean", "relu"],
            "cuda_kernel_activity": True,
        },
        "runtime_images": [
            {
                "basename": "libtorch_cpu.so",
                "wheel_relative_path": "libtorch_cpu.so",
                "sha256": "7" * 64,
                "size": 42,
                "build_id": "abcd",
                "ldd_match": True,
                "shared_by_torch_c_and_extension": True,
            }
        ],
    }


def test_canonical_envelope_round_trip(tmp_path: Path) -> None:
    envelope = make_envelope(_payload())
    verify_document(envelope)
    path = tmp_path / "evidence.json"
    path.write_bytes(canonical_json(envelope))
    assert len(verify_file(path)) == 64


def test_verifier_accepts_exact_unlinked_torch_global_deps_bootstrap() -> None:
    payload = _payload()
    payload["runtime_images"].append(  # type: ignore[union-attr]
        {
            "basename": "libtorch_global_deps.so",
            "wheel_relative_path": "libtorch_global_deps.so",
            "sha256": "8" * 64,
            "size": 84,
            "build_id": "dcba",
            "ldd_match": False,
            "shared_by_torch_c_and_extension": False,
        }
    )
    verify_document(make_envelope(payload))


@pytest.mark.parametrize("basename", ("libtorch_cpu.so", "libc10.so"))
def test_verifier_rejects_unlinked_nonbootstrap_runtime_image(basename: str) -> None:
    payload = _payload()
    images = payload["runtime_images"]
    assert isinstance(images, list)
    if basename == "libtorch_cpu.so":
        images[0]["basename"] = "libtorch_cuda.so"
        images[0]["wheel_relative_path"] = "libtorch_cuda.so"
    images.append(
        {
            "basename": basename,
            "wheel_relative_path": basename,
            "sha256": "8" * 64,
            "size": 84,
            "build_id": "dcba",
            "ldd_match": False,
            "shared_by_torch_c_and_extension": False,
        }
    )
    with pytest.raises(EvidenceError, match="not bound through ldd"):
        verify_document(make_envelope(payload))


@pytest.mark.parametrize(
    ("relative_path", "shared"),
    (("nested/libtorch_global_deps.so", False), ("libtorch_global_deps.so", True)),
)
def test_verifier_rejects_invalid_unlinked_torch_global_deps_identity(
    relative_path: str, shared: bool
) -> None:
    payload = _payload()
    payload["runtime_images"].append(  # type: ignore[union-attr]
        {
            "basename": "libtorch_global_deps.so",
            "wheel_relative_path": relative_path,
            "sha256": "8" * 64,
            "size": 84,
            "build_id": "dcba",
            "ldd_match": False,
            "shared_by_torch_c_and_extension": shared,
        }
    )
    with pytest.raises(EvidenceError, match="not bound through ldd"):
        verify_document(make_envelope(payload))


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("payload", "support_claim"), True),
        (("payload", "certification_ready"), True),
        (("payload", "kernel_executed"), False),
        (("payload", "device", "ordinal"), True),
        (("payload", "execution", "profiler_transfer_events"), ["aten::_to_copy"]),
        (("payload", "execution", "cuda_kernel_activity"), False),
        (("payload", "runtime_images", 0, "wheel_relative_path"), "../../host/libtorch.so"),
    ),
)
def test_verifier_rejects_overclaim_and_invariant_tampering(
    path: tuple[object, ...], value: object
) -> None:
    envelope = make_envelope(_payload())
    cursor: object = envelope
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    # Rebind the payload hash so this tests semantic checks, not only tamper detection.
    envelope = make_envelope(envelope["payload"])
    with pytest.raises(EvidenceError):
        verify_document(envelope)


def test_verifier_rejects_unknown_field_and_payload_hash_tamper() -> None:
    envelope = make_envelope(_payload())
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    payload["unknown"] = "no"
    with pytest.raises(EvidenceError, match="payload hash mismatch"):
        verify_document(envelope)
    rebound = make_envelope(envelope["payload"])
    with pytest.raises(EvidenceError, match="unknown or missing"):
        verify_document(rebound)


def test_verifier_rejects_machine_path_or_url_leak() -> None:
    for leaked in ("/home/owner/torch", "https://example.invalid/wheel"):
        payload = _payload()
        payload["device"]["name"] = leaked  # type: ignore[index]
        with pytest.raises(EvidenceError, match="path or URL"):
            verify_document(make_envelope(payload))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_scaled_error", 1.0001),
        ("max_scaled_error", float("inf")),
        ("max_scaled_error", True),
        ("numerically_equivalent", False),
    ),
)
def test_verifier_rejects_invalid_numerical_evidence(field: str, value: object) -> None:
    payload = _payload()
    payload["execution"][field] = value  # type: ignore[index]
    with pytest.raises(EvidenceError):
        verify_document(make_envelope(payload))


def test_verifier_binds_profile_hash_and_exact_rule_order() -> None:
    payload = _payload()
    payload["orchestration"]["provider_lock_artifact_profile_sha256"] = "c" * 64  # type: ignore[index]
    with pytest.raises(EvidenceError, match="artifact profiles differ"):
        verify_document(make_envelope(payload))
    payload = _payload()
    payload["artifact"]["ordered_rule_ids"] = list(  # type: ignore[index]
        reversed(payload["artifact"]["ordered_rule_ids"])  # type: ignore[index]
    )
    with pytest.raises(EvidenceError, match="ordered E2 rules"):
        verify_document(make_envelope(payload))


def test_extension_discovery_requires_one_nonempty_top_level_candidate(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    candidate = release / "libproject_rextio_native.so"
    candidate.write_bytes(b"native")
    (release / "deps").mkdir()
    (release / "deps/libdependency_rextio_native.so").write_bytes(b"ignored")
    assert _discover_extension(release) == candidate
    second = release / "other_rextio_native_abi.so"
    second.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="exactly one"):
        _discover_extension(release)


def test_direct_url_identity_never_hashes_url_or_requested_revision() -> None:
    first = {
        "url": "https://token-a@example.invalid/repo.git",
        "vcs_info": {
            "vcs": "git",
            "commit_id": "a" * 40,
            "requested_revision": "secret-branch-a",
        },
    }
    second = {
        "url": "https://token-b@example.invalid/other.git",
        "vcs_info": {
            "vcs": "git",
            "commit_id": "a" * 40,
            "requested_revision": "secret-branch-b",
        },
    }
    assert _sanitize_direct_url_identity(first) == _sanitize_direct_url_identity(second)


def test_checked_ldd_rejects_relevant_not_found() -> None:
    with pytest.raises(EvidenceError, match="libtorch_cuda.so"):
        _parse_ldd_checked("\tlibtorch_cuda.so => not found\n", "extension")


def test_runtime_ldd_prepends_exact_imported_torch_lib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch_package = tmp_path / "venv/site-packages/torch"
    torch_lib = torch_package / "lib"
    torch_lib.mkdir(parents=True)
    torch_init = torch_package / "__init__.py"
    torch_c = torch_package / "_C.so"
    extension = tmp_path / "rextio_native_abi.so"
    shared_image = torch_lib / "libtorch_cpu.so"
    for path in (torch_init, torch_c, extension, shared_image):
        path.write_bytes(b"fixture")

    torch = ModuleType("torch")
    torch.__file__ = str(torch_init)
    torch._C = ModuleType("torch._C")
    torch._C.__file__ = str(torch_c)

    guessed_cmake_lib = "/ambient/torch/share/cmake/lib"
    ambient = f"{guessed_cmake_lib}{os.pathsep}/usr/lib"
    monkeypatch.setenv("LD_LIBRARY_PATH", ambient)
    ldd_environments: list[dict[str, str] | None] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 1800,
    ) -> str:
        del cwd, timeout
        assert command[0] == "ldd"
        ldd_environments.append(env)
        return f"\tlibtorch_cpu.so => {shared_image} (0x7f00)\n"

    monkeypatch.setattr(certifier, "_run", fake_run)
    original_read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "" if path == Path("/proc/self/maps") else original_read_text(path, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        certifier,
        "parse_proc_maps",
        lambda _raw, root: {"libtorch_cpu.so": shared_image} if root == torch_lib else {},
    )
    monkeypatch.setattr(certifier, "_build_id", lambda _path: None)

    certifier._runtime_images(torch, extension)

    expected = f"{torch_lib}{os.pathsep}{ambient}"
    assert [env["LD_LIBRARY_PATH"] if env is not None else None for env in ldd_environments] == [
        expected,
        expected,
    ]


def test_runtime_images_retains_unlinked_torch_global_deps_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch_package = tmp_path / "venv/site-packages/torch"
    torch_lib = torch_package / "lib"
    torch_lib.mkdir(parents=True)
    torch_init = torch_package / "__init__.py"
    torch_c = torch_package / "_C.so"
    extension = tmp_path / "rextio_native_abi.so"
    shared_image = torch_lib / "libtorch_cpu.so"
    bootstrap_image = torch_lib / "libtorch_global_deps.so"
    for path in (torch_init, torch_c, extension, shared_image, bootstrap_image):
        path.write_bytes(b"fixture")

    torch = ModuleType("torch")
    torch.__file__ = str(torch_init)
    torch._C = ModuleType("torch._C")
    torch._C.__file__ = str(torch_c)

    monkeypatch.setattr(
        certifier,
        "_run",
        lambda command, **_kwargs: (
            f"\tlibtorch_cpu.so => {shared_image} (0x7f00)\n"
            if command[0] == "ldd"
            else ""
        ),
    )
    original_read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "" if path == Path("/proc/self/maps") else original_read_text(path, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        certifier,
        "parse_proc_maps",
        lambda _raw, root: {
            "libtorch_cpu.so": shared_image,
            "libtorch_global_deps.so": bootstrap_image,
        }
        if root == torch_lib
        else {},
    )
    monkeypatch.setattr(certifier, "_build_id", lambda path: "dcba" if path == bootstrap_image else None)

    rows = certifier._runtime_images(torch, extension)

    assert rows == [
        {
            "basename": "libtorch_cpu.so",
            "wheel_relative_path": "libtorch_cpu.so",
            "sha256": certifier.sha256_file(shared_image),
            "size": shared_image.stat().st_size,
            "build_id": None,
            "ldd_match": True,
            "shared_by_torch_c_and_extension": True,
        },
        {
            "basename": "libtorch_global_deps.so",
            "wheel_relative_path": "libtorch_global_deps.so",
            "sha256": certifier.sha256_file(bootstrap_image),
            "size": bootstrap_image.stat().st_size,
            "build_id": "dcba",
            "ldd_match": False,
            "shared_by_torch_c_and_extension": False,
        },
    ]


def test_module_identity_must_resolve_under_exact_checkout(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    package = root / "src/package"
    package.mkdir(parents=True)
    module_file = package / "__init__.py"
    module_file.write_text("", encoding="utf-8")
    module = ModuleType("fixture")
    module.__file__ = str(module_file)
    _require_module_under_root(module, root, "fixture")
    outside = tmp_path / "site-packages/package.py"
    outside.parent.mkdir()
    outside.write_text("", encoding="utf-8")
    module.__file__ = str(outside)
    with pytest.raises(RuntimeError, match="exact checkout"):
        _require_module_under_root(module, root, "fixture")


@pytest.mark.parametrize(
    "command",
    (
        ("scripts/certify_cuda_candidate.py", "--help"),
        ("-m", "scripts.certify_cuda_candidate", "--help"),
    ),
)
def test_certification_cli_help_supports_script_and_module_invocation(
    command: tuple[str, ...],
) -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), str(root), env.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, *command],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--expected-torch-commit" in completed.stdout


def test_cuda_graph_replay_is_nested_in_selected_stream_context() -> None:
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse(
        (root / "scripts/certify_cuda_candidate.py").read_text(encoding="utf-8")
    )
    capture = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_capture_and_profile"
    )
    stream_blocks = [
        node
        for node in ast.walk(capture)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "stream"
            and len(item.context_expr.args) == 1
            and isinstance(item.context_expr.args[0], ast.Name)
            and item.context_expr.args[0].id == "stream"
            for item in node.items
        )
    ]
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "graph"
        and call.func.attr == "replay"
        for block in stream_blocks
        for call in ast.walk(block)
        if isinstance(call, ast.Call)
    )


def test_file_verifier_rejects_noncanonical_json(tmp_path: Path) -> None:
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(make_envelope(_payload()), indent=2), encoding="utf-8")
    with pytest.raises(EvidenceError, match="canonically"):
        verify_file(path)


def test_proc_maps_requires_one_wheel_rooted_image_per_basename(tmp_path: Path) -> None:
    torch_lib = tmp_path / "torch/lib"
    torch_lib.mkdir(parents=True)
    cpu = torch_lib / "libtorch_cpu.so"
    c10 = torch_lib / "libc10.so"
    cpu.write_bytes(b"cpu")
    c10.write_bytes(b"c10")
    text = (
        f"7f00-7f01 r-xp 00000000 00:00 1 {cpu}\n"
        f"7f02-7f03 r--p 00000000 00:00 1 {cpu}\n"
        f"7f04-7f05 r-xp 00000000 00:00 2 {c10}\n"
    )
    parsed = parse_proc_maps(text, torch_lib)
    assert parsed == {"libc10.so": c10, "libtorch_cpu.so": cpu}
    outside = tmp_path / "outside/libtorch_cpu.so"
    outside.parent.mkdir()
    outside.write_bytes(b"other")
    with pytest.raises(EvidenceError, match="outside torch/lib"):
        parse_proc_maps(text + f"7f06-7f07 r-xp 0 00:00 3 {outside}\n", torch_lib)


def test_ldd_parser_and_profiler_transfer_filter() -> None:
    parsed = parse_ldd(
        "\tlibtorch_cpu.so => /wheel/torch/lib/libtorch_cpu.so (0x7f00)\n"
        "\tlinux-vdso.so.1 (0x7fff)\n"
    )
    assert parsed == {"libtorch_cpu.so": Path("/wheel/torch/lib/libtorch_cpu.so")}
    assert forbidden_profiler_events(
        ["aten::mm", "Memcpy HtoD (Pageable -> Device)", "aten::_to_copy"]
    ) == ["Memcpy HtoD (Pageable -> Device)", "aten::_to_copy"]


def test_redact_path_never_emits_absolute_or_parent_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    child = root / "lib.so"
    child.write_bytes(b"x")
    assert redact_path(child, root) == "lib.so"
    outside = tmp_path / "outside.so"
    outside.write_bytes(b"x")
    with pytest.raises(EvidenceError, match="outside"):
        redact_path(outside, root)
