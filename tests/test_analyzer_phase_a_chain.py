"""Analyzer integration: real-source Phase A linear→ReLU→mean chain.

Uses Core API 1.6 ``analyze_project`` (same harness shape as
rextio-numpy/pandas analyzer integration tests). Does not hand-construct
``ClaimSite`` objects — claims must come from nested AST inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rextio.analyzer.models import FunctionAnalysis, ProjectAnalysis
from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_torch.claim.activations import RELU_RULE
from rextio_torch.claim.linear import LINEAR_RULE, LINEAR_TARGET
from rextio_torch.claim.reductions import MEAN_RULE
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.plugin import PLUGIN_ID, plugin

PHASE_A_SOURCE = '''
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch.nn.functional as F


def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
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


def test_analyzer_discovers_nested_phase_a_claims_and_native_route(tmp_path: Path) -> None:
    root = _write_module(tmp_path, PHASE_A_SOURCE)
    registry = _registry()
    assert registry.active[0].api_version == "1.6"
    assert registry.active[0].lowering_provided is True

    analysis = analyze_project(
        root,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )

    inference = _function(analysis, "myapp.kernels.inference")
    assert inference.accepted is True
    assert inference.route == f"native-plugin:{PLUGIN_ID}"
    assert not any(d.severity == "error" for d in inference.diagnostics)

    claims = list(inference.plugin_claims)
    assert len(claims) == 3

    by_rule = {claim.rule_id: claim for claim in claims}
    assert set(by_rule) == {LINEAR_RULE, RELU_RULE, MEAN_RULE}

    linear = by_rule[LINEAR_RULE]
    assert linear.target == LINEAR_TARGET
    assert linear.operand_types == (
        TENSOR_F32_CPU_2D,
        TENSOR_F32_CPU_2D,
        TENSOR_F32_CPU_1D,
    )
    assert linear.receiver is None
    assert linear.keywords == ()
    assert linear.result_type == TENSOR_F32_CPU_2D

    relu = by_rule[RELU_RULE]
    assert relu.target.rpartition(".")[2] == "relu"
    assert relu.receiver is not None
    assert relu.receiver.arg_type == TENSOR_F32_CPU_2D
    assert relu.receiver.expr_kind == "call"
    assert relu.receiver.is_safe is False
    assert relu.operand_types == ()
    assert relu.keywords == ()
    assert relu.result_type == TENSOR_F32_CPU_2D

    mean = by_rule[MEAN_RULE]
    assert mean.target.rpartition(".")[2] == "mean"
    assert mean.receiver is not None
    assert mean.receiver.arg_type == TENSOR_F32_CPU_2D
    assert mean.receiver.expr_kind == "call"
    assert mean.receiver.is_safe is False
    assert mean.operand_types == ()
    assert {kw.name: kw.literal.value for kw in mean.keywords} == {
        "dim": 1,
        "keepdim": False,
    }
    # Rank-2 receiver → rank-1 result after dim=1 reduction.
    assert mean.result_type == TENSOR_F32_CPU_1D

    # Nested claim order follows Python evaluation: linear, then relu, then mean.
    assert [claim.rule_id for claim in claims] == [LINEAR_RULE, RELU_RULE, MEAN_RULE]
