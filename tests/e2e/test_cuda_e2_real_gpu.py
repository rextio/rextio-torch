"""Explicit opt-in entry point for the manual real-NVIDIA candidate."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_cuda_gpu


def test_manual_cuda_e2_candidate() -> None:
    """Run only when the owner supplies the complete explicit environment."""
    if os.environ.get("REXTIO_RUN_CUDA_E2") != "1":
        pytest.skip("set REXTIO_RUN_CUDA_E2=1 to run the real NVIDIA candidate")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        pytest.skip("real NVIDIA candidate is Linux x86_64 only")
    required = {
        "REXTIO_CUDA_E2_SM",
        "REXTIO_CUDA_E2_EXPECTED_TORCH_COMMIT",
        "REXTIO_CUDA_E2_OUTPUT",
        "REXTIO_CUDA_E2_WORK_DIR",
    }
    missing = sorted(required - os.environ.keys())
    assert not missing, f"missing explicit CUDA E2 variables: {missing}"
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_cuda_candidate.py"),
            "--sm",
            os.environ["REXTIO_CUDA_E2_SM"],
            "--expected-torch-commit",
            os.environ["REXTIO_CUDA_E2_EXPECTED_TORCH_COMMIT"],
            "--output",
            os.environ["REXTIO_CUDA_E2_OUTPUT"],
            "--work-dir",
            os.environ["REXTIO_CUDA_E2_WORK_DIR"],
        ],
        cwd=root,
        env=os.environ,
        check=False,
    )
    assert completed.returncode == 0
