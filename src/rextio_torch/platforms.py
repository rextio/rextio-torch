"""Declarative host-platform contract for the 0.1.0 public Alpha.

This module never probes the ambient machine.  Callers provide a normalized
OS/architecture pair so CI, packaging tools, and future Rextio target
resolution can reason about unavailable profiles without pretending to run
them.  Native execution remains governed by the exact CPython/PyTorch/tch
runtime checks documented in the README.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

HostStatus = Literal["certified", "experimental", "availability-gated", "unsupported"]
NativeE2E = Literal["required", "experimental", "unavailable"]


@dataclass(frozen=True, slots=True)
class HostProfile:
    """One explicit Alpha host-platform disposition."""

    os: Literal["linux", "macos"]
    arch: Literal["x86", "x64", "arm32", "arm64"]
    status: HostStatus
    native_e2e: NativeE2E
    reason_code: str
    detail: str

    @property
    def key(self) -> str:
        """Return the stable matrix key."""
        return f"{self.os}-{self.arch}"


class NativeRuntimeUnavailable(RuntimeError):
    """The requested profile has no certifiable pinned native runtime."""


HOST_PROFILES: tuple[HostProfile, ...] = (
    HostProfile(
        "linux",
        "x86",
        "unsupported",
        "unavailable",
        "RXTT-PLATFORM-TORCH-WHEEL-UNAVAILABLE",
        "torch 2.11.0 has no CPython 3.11 Linux i686 wheel",
    ),
    HostProfile(
        "linux",
        "x64",
        "experimental",
        "required",
        "RXTT-PLATFORM-ALPHA-EXPERIMENTAL",
        "hosted runner and pinned manylinux x86_64 wheel are available",
    ),
    HostProfile(
        "linux",
        "arm32",
        "unsupported",
        "unavailable",
        "RXTT-PLATFORM-TORCH-WHEEL-UNAVAILABLE",
        "torch 2.11.0 has no CPython 3.11 Linux ARMv7 wheel",
    ),
    HostProfile(
        "linux",
        "arm64",
        "experimental",
        "experimental",
        "RXTT-PLATFORM-ALPHA-EXPERIMENTAL",
        "pinned manylinux AArch64 wheel exists; hosted-runner evidence is non-blocking",
    ),
    HostProfile(
        "macos",
        "x86",
        "unsupported",
        "unavailable",
        "RXTT-PLATFORM-MODERN-TARGET-UNAVAILABLE",
        "modern GitHub-hosted macOS i686 runners and torch wheels do not exist",
    ),
    HostProfile(
        "macos",
        "x64",
        "availability-gated",
        "unavailable",
        "RXTT-PLATFORM-TORCH-WHEEL-UNAVAILABLE",
        "torch 2.11.0 has no CPython 3.11 macOS x86_64 wheel",
    ),
    HostProfile(
        "macos",
        "arm32",
        "unsupported",
        "unavailable",
        "RXTT-PLATFORM-MODERN-TARGET-UNAVAILABLE",
        "modern GitHub-hosted macOS ARMv7 runners and torch wheels do not exist",
    ),
    HostProfile(
        "macos",
        "arm64",
        "certified",
        "required",
        "RXTT-PLATFORM-CERTIFIED",
        "certified CPython 3.11 / torch 2.11.0 / tch 0.24.0 profile",
    ),
)

_PROFILE_BY_KEY = {profile.key: profile for profile in HOST_PROFILES}


def host_profile(os_name: str, arch: str) -> HostProfile:
    """Return an explicitly listed profile, rejecting unknown spellings."""
    key = f"{os_name}-{arch}"
    try:
        return _PROFILE_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"rextio-torch: unknown Alpha host profile {key!r}") from exc


def require_native_runtime(os_name: str, arch: str) -> HostProfile:
    """Require a runtime-backed cell or fail closed with a stable reason."""
    profile = host_profile(os_name, arch)
    if profile.native_e2e == "unavailable":
        raise NativeRuntimeUnavailable(
            f"rextio-torch: {profile.key} native runtime unavailable "
            f"[{profile.reason_code}]: {profile.detail}"
        )
    return profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one rextio-torch Alpha host cell")
    parser.add_argument("--os", choices=("linux", "macos"), required=True)
    parser.add_argument("--arch", choices=("x86", "x64", "arm32", "arm64"), required=True)
    parser.add_argument(
        "--expect",
        choices=("certified", "experimental", "availability-gated", "unsupported"),
        required=True,
    )
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="also require a runtime-backed E2E profile",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate a CI matrix cell and return a process status."""
    args = _parser().parse_args(argv)
    profile = host_profile(args.os, args.arch)
    if profile.status != args.expect:
        raise SystemExit(
            f"rextio-torch: {profile.key} expected {args.expect!r}, "
            f"contract says {profile.status!r}"
        )
    if args.require_native:
        require_native_runtime(args.os, args.arch)
    print(
        f"{profile.key}: status={profile.status} native_e2e={profile.native_e2e} "
        f"reason={profile.reason_code}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by Actions
    raise SystemExit(main())
