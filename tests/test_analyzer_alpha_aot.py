"""Analyzer integration for the broader Alpha AOT surface (no real Cargo)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rextio.analyzer.models import FunctionAnalysis, ProjectAnalysis
from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_torch.claim.activations import RELU_RULE, SIGMOID_RULE_2D
from rextio_torch.claim.binops import (
    ADD_BROADCAST_2D_1D_RULE,
    MATMUL_BINOP_RULE,
    MATMUL_CALL_RULE,
)
from rextio_torch.claim.reductions import MEAN_RULE
from rextio_torch.diagnostics import TENSOR_F32_CPU_2D
from rextio_torch.plugin import PLUGIN_ID, plugin

# Temps are required: core does not claim method calls whose receiver is a bare BinOp
# (e.g. ``(a @ b + bias).relu()``). Named receivers and call-chain receivers work.
ALPHA_CONTROL_SOURCE = '''
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch


def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
    depth: int,
    phase: int,
) -> TensorF32Cpu1D:
    hidden = x
    for layer in range(depth):
        if (layer + phase) % 2 == 0:
            even = hidden @ weight + bias
            hidden = even.relu()
        else:
            odd = hidden @ weight + bias
            hidden = odd.sigmoid()
    return hidden.mean(dim=1, keepdim=False)


def functional_matmul(
    left: TensorF32Cpu2D,
    right: TensorF32Cpu2D,
) -> TensorF32Cpu2D:
    return torch.matmul(left, right)


def method_matmul(
    left: TensorF32Cpu2D,
    right: TensorF32Cpu2D,
) -> TensorF32Cpu2D:
    return left.matmul(right)
'''


class FakeEntryPoint:
    name = PLUGIN_ID

    def load(self) -> Any:
        return plugin


def _write_module(root: Path, body: str) -> Path:
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-torch"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "myapp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(body, encoding="utf-8")
    return root


def _registry():
    return load_plugin_registry(
        PluginConfig(enabled=(PLUGIN_ID,)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )


def _function(analysis: ProjectAnalysis, qualname: str) -> FunctionAnalysis:
    for module in analysis.modules:
        for function in module.functions:
            if function.qualname == qualname:
                return function
    raise AssertionError(f"function {qualname!r} not found")


def test_analyzer_alpha_control_flow_routes_native(tmp_path: Path) -> None:
    registry = _registry()
    analysis = analyze_project(
        _write_module(tmp_path, ALPHA_CONTROL_SOURCE),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    inference = _function(analysis, "myapp.kernels.inference")
    assert inference.accepted is True
    assert inference.route == f"native-plugin:{PLUGIN_ID}"
    assert not any(d.severity == "error" for d in inference.diagnostics)

    rule_ids = [claim.rule_id for claim in inference.plugin_claims]
    assert MATMUL_BINOP_RULE in rule_ids
    assert ADD_BROADCAST_2D_1D_RULE in rule_ids
    assert RELU_RULE in rule_ids
    assert SIGMOID_RULE_2D in rule_ids
    assert MEAN_RULE in rule_ids
    # Both branches of the if are claimed (matmul/add/act each appear twice).
    assert rule_ids.count(MATMUL_BINOP_RULE) == 2
    assert rule_ids.count(RELU_RULE) == 1
    assert rule_ids.count(SIGMOID_RULE_2D) == 1

    for qualname in ("functional_matmul", "method_matmul"):
        function = _function(analysis, f"myapp.kernels.{qualname}")
        assert function.accepted is True
        assert function.route == f"native-plugin:{PLUGIN_ID}"
        assert not any(d.severity == "error" for d in function.diagnostics)
        assert len(function.plugin_claims) == 1
        claim = function.plugin_claims[0]
        assert claim.rule_id == MATMUL_CALL_RULE
        assert claim.kind == "call"
        assert claim.result_type == TENSOR_F32_CPU_2D

    functional = _function(analysis, "myapp.kernels.functional_matmul")
    assert functional.plugin_claims[0].target == "torch.matmul"
    assert functional.plugin_claims[0].receiver is None

    method = _function(analysis, "myapp.kernels.method_matmul")
    assert method.plugin_claims[0].target.rpartition(".")[2] == "matmul"
    assert method.plugin_claims[0].receiver is not None
    assert method.plugin_claims[0].receiver.arg_type == TENSOR_F32_CPU_2D
