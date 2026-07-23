"""Analyzer integration for exact default/explicit-none functional GELU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rextio.analyzer.models import FunctionAnalysis, ProjectAnalysis
from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_torch.claim.gelu import GELU_NONE_RULE
from rextio_torch.plugin import PLUGIN_ID, plugin

SOURCE = """
from rextio_torch.types import TensorF32Cpu2D
import torch
import torch.nn.functional as F


def gelu_default(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return F.gelu(x)


def gelu_explicit_none(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return F.gelu(x, approximate="none")


def gelu_tanh(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return F.gelu(x, approximate="tanh")


def gelu_dynamic(x: TensorF32Cpu2D, mode: str) -> TensorF32Cpu2D:
    return F.gelu(x, approximate=mode)


def gelu_positional_option(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return F.gelu(x, "none")


def gelu_module(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    layer = torch.nn.GELU()
    return layer(x)
"""


class FakeEntryPoint:
    name = PLUGIN_ID

    def load(self) -> Any:
        return plugin


def _registry():
    return load_plugin_registry(
        PluginConfig(enabled=(PLUGIN_ID,)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )


def _write_project(root: Path) -> Path:
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-torch"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "myapp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(SOURCE, encoding="utf-8")
    return root


def _function(analysis: ProjectAnalysis, name: str) -> FunctionAnalysis:
    qualname = f"myapp.kernels.{name}"
    for module in analysis.modules:
        for function in module.functions:
            if function.qualname == qualname:
                return function
    raise AssertionError(f"function {qualname!r} not found")


def test_analyzer_routes_only_exact_none_gelu_forms(tmp_path: Path) -> None:
    registry = _registry()
    analysis = analyze_project(
        _write_project(tmp_path),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )

    for name in ("gelu_default", "gelu_explicit_none"):
        function = _function(analysis, name)
        assert function.accepted is True
        assert function.route == f"native-plugin:{PLUGIN_ID}"
        assert [claim.rule_id for claim in function.plugin_claims] == [GELU_NONE_RULE]
        claim = function.plugin_claims[0]
        assert claim.target == "torch.nn.functional.gelu"
        assert claim.result_type == "rextio-torch/tensor-f32-cpu-2d"

    for name in ("gelu_tanh", "gelu_dynamic", "gelu_positional_option", "gelu_module"):
        function = _function(analysis, name)
        assert function.route != f"native-plugin:{PLUGIN_ID}"
        assert not function.plugin_claims
