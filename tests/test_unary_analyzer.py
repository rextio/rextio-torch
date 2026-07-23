"""Analyzer integration for exact bounded unary math spellings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rextio.analyzer.models import FunctionAnalysis, ProjectAnalysis
from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_torch.claim.unary import (
    UNARY_ABS_RULE,
    UNARY_EXP_RULE,
    UNARY_LOG_RULE,
    UNARY_NEGATIVE_RULE,
    UNARY_NEG_RULE,
    UNARY_SQRT_RULE,
    UNARY_SQUARE_RULE,
)
from rextio_torch.plugin import PLUGIN_ID, plugin

SOURCE = """
from rextio_torch.types import TensorF32Cpu2D
import torch


def unary_function_surface(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    absolute = torch.abs(x)
    negated = torch.neg(absolute)
    negative = torch.negative(negated)
    squared = torch.square(negative)
    exponent = torch.exp(squared)
    logged = torch.log(exponent)
    return torch.sqrt(logged)


def unary_method_surface(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    absolute = x.abs()
    negated = absolute.neg()
    negative = negated.negative()
    squared = negative.square()
    exponent = squared.exp()
    logged = exponent.log()
    return logged.sqrt()


def invalid_unary_out(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return torch.abs(x, out=x)


def invalid_unary_method_arg(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return x.sqrt(x)


def noncanonical_absolute(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return torch.absolute(x)
"""

EXPECTED_RULES = [
    UNARY_ABS_RULE,
    UNARY_NEG_RULE,
    UNARY_NEGATIVE_RULE,
    UNARY_SQUARE_RULE,
    UNARY_EXP_RULE,
    UNARY_LOG_RULE,
    UNARY_SQRT_RULE,
]


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


def test_analyzer_routes_exact_unary_function_and_method_surfaces(tmp_path: Path) -> None:
    registry = _registry()
    analysis = analyze_project(
        _write_project(tmp_path),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )

    for name in ("unary_function_surface", "unary_method_surface"):
        function = _function(analysis, name)
        assert function.accepted is True
        assert function.route == f"native-plugin:{PLUGIN_ID}"
        assert [claim.rule_id for claim in function.plugin_claims] == EXPECTED_RULES
        assert all(
            claim.result_type == "rextio-torch/tensor-f32-cpu-2d"
            for claim in function.plugin_claims
        )

    for name in ("invalid_unary_out", "invalid_unary_method_arg", "noncanonical_absolute"):
        function = _function(analysis, name)
        assert function.route != f"native-plugin:{PLUGIN_ID}"
        assert not function.plugin_claims
