"""Plugin discovery and types must not import torch or heavy core hosts."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _write_minimal_rextio(root: Path) -> Path:
    runtime_python = root / "build" / "python"
    rextio_pkg = runtime_python / "rextio"
    rextio_pkg.mkdir(parents=True)
    (rextio_pkg / "__init__.py").write_text(
        '"""Minimal generated-runtime rextio package."""\n',
        encoding="utf-8",
    )
    (rextio_pkg / "__about__.py").write_text(
        '__version__ = "0.0.0-generated-runtime"\n',
        encoding="utf-8",
    )
    (rextio_pkg / "runtime.py").write_text(
        '"""Generated runtime helpers stub."""\n',
        encoding="utf-8",
    )
    return runtime_python


def test_types_and_root_import_without_torch_or_rextio_config(tmp_path: Path) -> None:
    runtime_python = _write_minimal_rextio(tmp_path)
    script = textwrap.dedent(
        """\
        import importlib.util
        import sys

        assert importlib.util.find_spec("rextio") is not None
        assert importlib.util.find_spec("rextio.config") is None
        assert importlib.util.find_spec("rextio.plugins") is None

        import rextio_torch.types as types
        assert types.TensorF32Cpu2D is not None
        assert types.TensorF32Cpu1D is not None
        assert types.TensorF32Cuda0_2D is not None
        assert types.TensorF32Cuda0_1D is not None

        from rextio_torch import RextioTorchPlugin, __version__, plugin
        assert isinstance(__version__, str) and __version__
        assert callable(plugin)
        provider = plugin()
        assert isinstance(provider, RextioTorchPlugin)
        assert provider.plugin_id == "rextio-torch"
        assert provider.api_version == "1.6"

        assert "torch" not in sys.modules
        print("ok")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(runtime_python), str(SRC_ROOT), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_plugin_module_source_does_not_import_torch() -> None:
    import re

    plugin_src = (SRC_ROOT / "rextio_torch" / "plugin.py").read_text(encoding="utf-8")
    types_src = (SRC_ROOT / "rextio_torch" / "types.py").read_text(encoding="utf-8")
    torch_import = re.compile(
        r"(?m)^\s*(import\s+torch(\s|$|\.)|from\s+torch(\s|\.))"
    )
    assert torch_import.search(plugin_src) is None
    assert torch_import.search(types_src) is None
