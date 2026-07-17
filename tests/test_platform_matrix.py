"""Truth-model tests for every requested Linux/macOS architecture cell."""

from __future__ import annotations

import pytest

from rextio_torch.platforms import (
    HOST_PROFILES,
    NativeRuntimeUnavailable,
    host_profile,
    main,
    require_native_runtime,
)


def test_matrix_is_complete_and_unique() -> None:
    expected = {
        f"{os_name}-{arch}"
        for os_name in ("linux", "macos")
        for arch in ("x86", "x64", "arm32", "arm64")
    }
    keys = [profile.key for profile in HOST_PROFILES]
    assert set(keys) == expected
    assert len(keys) == len(set(keys)) == 8


@pytest.mark.parametrize(
    ("os_name", "arch", "status", "native_e2e"),
    [
        ("linux", "x86", "unsupported", "unavailable"),
        ("linux", "x64", "experimental", "required"),
        ("linux", "arm32", "unsupported", "unavailable"),
        ("linux", "arm64", "experimental", "experimental"),
        ("macos", "x86", "unsupported", "unavailable"),
        ("macos", "x64", "availability-gated", "unavailable"),
        ("macos", "arm32", "unsupported", "unavailable"),
        ("macos", "arm64", "certified", "required"),
    ],
)
def test_each_requested_cell_has_an_honest_disposition(
    os_name: str,
    arch: str,
    status: str,
    native_e2e: str,
) -> None:
    profile = host_profile(os_name, arch)
    assert profile.status == status
    assert profile.native_e2e == native_e2e
    assert profile.reason_code.startswith("RXTT-PLATFORM-")
    assert profile.detail


@pytest.mark.parametrize(
    ("os_name", "arch", "reason"),
    [
        ("linux", "x86", "RXTT-PLATFORM-TORCH-WHEEL-UNAVAILABLE"),
        ("linux", "arm32", "RXTT-PLATFORM-TORCH-WHEEL-UNAVAILABLE"),
        ("macos", "x86", "RXTT-PLATFORM-MODERN-TARGET-UNAVAILABLE"),
        ("macos", "arm32", "RXTT-PLATFORM-MODERN-TARGET-UNAVAILABLE"),
    ],
)
def test_32_bit_and_impossible_cells_fail_closed(
    os_name: str,
    arch: str,
    reason: str,
) -> None:
    with pytest.raises(NativeRuntimeUnavailable, match=reason):
        require_native_runtime(os_name, arch)


def test_macos_x64_is_availability_gated_not_supported() -> None:
    profile = host_profile("macos", "x64")
    assert profile.status == "availability-gated"
    with pytest.raises(
        NativeRuntimeUnavailable,
        match="RXTT-PLATFORM-TORCH-WHEEL-UNAVAILABLE",
    ):
        require_native_runtime("macos", "x64")


def test_runtime_backed_profiles_are_explicit() -> None:
    assert require_native_runtime("macos", "arm64").status == "certified"
    assert require_native_runtime("linux", "x64").native_e2e == "required"
    assert require_native_runtime("linux", "arm64").native_e2e == "experimental"


def test_unknown_platform_spelling_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown Alpha host profile"):
        host_profile("darwin", "aarch64")


def test_cli_contract_check() -> None:
    assert main(["--os", "linux", "--arch", "x86", "--expect", "unsupported"]) == 0
    with pytest.raises(SystemExit, match="contract says 'unsupported'"):
        main(["--os", "linux", "--arch", "x86", "--expect", "certified"])
