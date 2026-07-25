#!/usr/bin/env bash
# Experimental Linux smoke for the rextio-torch 0.1.2 Alpha.
#
# Exercises the pinned native-AOT contract on Linux x86_64 or aarch64.
# This is NOT certification. Certified host remains macOS arm64.
# Windows is deferred/unverified — do not treat this script as a Windows path.
#
# Usage:
#   ./scripts/linux-smoke.sh           # focused unit tests only
#   ./scripts/linux-smoke.sh --cargo   # also run real-Cargo e2e when cargo exists
#
# Required env contract:
#   - CPython 3.11
#   - torch==2.11.0 on the active interpreter
#   - LIBTORCH_USE_PYTORCH=1
#   - LIBTORCH_BYPASS_VERSION_CHECK must NOT be set

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_CARGO=0
for arg in "$@"; do
  case "$arg" in
    --cargo) RUN_CARGO=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      echo "usage: $0 [--cargo]" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${VIRTUAL_ENV:-}/bin/python" && ! -x ".venv/bin/python" ]]; then
  echo "error: need an active venv or ./.venv with CPython 3.11 + editable install" >&2
  echo "hint: follow README.md 'Linux experimental verification recipe' (exact Core commit, then this repo --no-deps)" >&2
  exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
  # Prefer venv bin first so torch-sys's bare `python` resolves correctly.
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
else
  PYTHON="$(cd "$ROOT" && pwd)/.venv/bin/python"
  export VIRTUAL_ENV="$(cd "$ROOT" && pwd)/.venv"
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

export PYO3_PYTHON="$PYTHON"
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK || true
if [[ -n "${LIBTORCH_BYPASS_VERSION_CHECK+x}" ]]; then
  echo "error: LIBTORCH_BYPASS_VERSION_CHECK must not be set" >&2
  exit 1
fi

echo "== host =="
uname -sm || true
"$PYTHON" - <<'PY'
import platform
import sys

system = platform.system()
machine = platform.machine()
print(f"python={sys.version.split()[0]} ({sys.version_info[:2]})")
print(f"system={system} machine={machine}")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"error: CPython 3.11 required, got {sys.version_info[:3]}")
if system == "Windows":
    raise SystemExit(
        "error: Windows is deferred/unverified for this Alpha; "
        "linux-smoke.sh is for Linux experimental hosts (or pin checks on macOS)"
    )
if system == "Linux":
    print("role=experimental Linux smoke (NOT certification)")
elif system == "Darwin":
    print(
        "role=pin-contract check on macOS "
        "(certified host path; does not produce Linux evidence)"
    )
else:
    raise SystemExit(
        f"error: unsupported host system {system!r}; "
        "linux-smoke.sh accepts Linux (experimental) or Darwin (pin check) only"
    )
PY

echo "== torch pin (exact 2.11.0; no version-check bypass) =="
"$PYTHON" - <<'PY'
import torch

version = torch.__version__.split("+")[0]
print(f"torch={torch.__version__}")
if version != "2.11.0":
    raise SystemExit(
        f"error: torch==2.11.0 required for tch 0.24.0 / libtorch contract; got {version!r}"
    )
PY

echo "== rextio plugin API compatibility =="
"$PYTHON" - <<'PY'
from rextio.plugins.api import PLUGIN_API_VERSION
from rextio_torch.plugin import REQUIRED_PLUGIN_API, plugin

print(f"PLUGIN_API_VERSION={PLUGIN_API_VERSION}")
print(f"REQUIRED_PLUGIN_API={REQUIRED_PLUGIN_API}")
provider = plugin()
deps = provider.crate_dependencies()
assert deps[0].name == "tch" and deps[0].version == "=0.24.0"
assert deps[0].features == ("python-extension",)
print("crate pin: tch =0.24.0 features=python-extension")
PY

echo "== focused unit tests (no e2e) =="
"$PYTHON" -m pytest -q tests --ignore=tests/e2e

if [[ "$RUN_CARGO" -eq 1 ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "error: --cargo requested but cargo is not on PATH" >&2
    echo "install a Rust toolchain or omit --cargo" >&2
    exit 1
  fi
  echo "== real-Cargo e2e (experimental; fail-closed on pin/toolchain errors) =="
  echo "cargo=$(command -v cargo)"
  rustc --version || true
  # needs_cargo tests skip only when cargo is missing; with cargo present they
  # must surface torch/libtorch/ABI failures (never via BYPASS_VERSION_CHECK).
  "$PYTHON" -m pytest -q tests/e2e -m needs_cargo
else
  echo "== skipping real-Cargo e2e (pass --cargo to enable) =="
fi

echo
echo "linux-smoke: OK (experimental evidence only; certified host remains macOS arm64)"
