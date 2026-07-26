"""Analyzer integration for the diagnostic small-batch scoring kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rextio.analyzer.models import FunctionAnalysis, ProjectAnalysis
from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from benchmarks.small_batch_scoring_cases import (
    EXPECTED_CLAIM_RULE_FAMILIES,
    KERNEL_SOURCE,
)
from rextio_torch.plugin import PLUGIN_ID, plugin


class FakeEntryPoint:
    name = PLUGIN_ID

    def load(self) -> Any:
        return plugin


def _write_module(root: Path) -> Path:
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-torch"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "myapp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNEL_SOURCE, encoding="utf-8")
    return root


def _function(analysis: ProjectAnalysis, qualname: str) -> FunctionAnalysis:
    for module in analysis.modules:
        for function in module.functions:
            if function.qualname == qualname:
                return function
    raise AssertionError(f"function {qualname!r} not found")


def _analyze(tmp_path: Path) -> ProjectAnalysis:
    registry = load_plugin_registry(
        PluginConfig(enabled=(PLUGIN_ID,)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )
    return analyze_project(
        _write_module(tmp_path),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )


def test_score_logits_routes_native_with_control_flow_claims(tmp_path: Path) -> None:
    analysis = _analyze(tmp_path)
    logits = _function(analysis, "myapp.kernels.score_logits")
    assert logits.accepted is True
    assert logits.route == f"native-plugin:{PLUGIN_ID}"
    assert not any(diagnostic.severity == "error" for diagnostic in logits.diagnostics)
    rules = [claim.rule_id for claim in logits.plugin_claims]
    assert "rextio-torch/tensor-sub-f32-cpu-2d-1d-broadcast" in rules
    assert "rextio-torch/tensor-div-f32-cpu-2d-1d-broadcast" in rules
    assert rules.count("rextio-torch/functional-linear-f32-cpu-2d") == 2
    assert "rextio-torch/tensor-relu-f32-cpu-2d" in rules
    assert "rextio-torch/tensor-tanh-f32-cpu-2d" in rules
    assert "rextio-torch/tensor-softmax-dim1-f32-cpu-2d" not in rules
    assert "rextio-torch/tensor-argmax-dim1-keepfalse-i64-cpu-1d" not in rules


def test_score_probabilities_and_labels_complete_the_head(tmp_path: Path) -> None:
    analysis = _analyze(tmp_path)
    probs = _function(analysis, "myapp.kernels.score_probabilities")
    labels = _function(analysis, "myapp.kernels.score_labels")
    for function in (probs, labels):
        assert function.accepted is True
        assert function.route == f"native-plugin:{PLUGIN_ID}"
        assert not any(diagnostic.severity == "error" for diagnostic in function.diagnostics)

    prob_rules = [claim.rule_id for claim in probs.plugin_claims]
    label_rules = [claim.rule_id for claim in labels.plugin_claims]
    assert "rextio-torch/tensor-softmax-dim1-f32-cpu-2d" in prob_rules
    assert "rextio-torch/tensor-argmax-dim1-keepfalse-i64-cpu-1d" not in prob_rules
    assert "rextio-torch/tensor-softmax-dim1-f32-cpu-2d" in label_rules
    assert "rextio-torch/tensor-argmax-dim1-keepfalse-i64-cpu-1d" in label_rules

    observed = set(prob_rules) | set(label_rules) | {
        claim.rule_id
        for claim in _function(analysis, "myapp.kernels.score_logits").plugin_claims
    }
    assert EXPECTED_CLAIM_RULE_FAMILIES <= observed
