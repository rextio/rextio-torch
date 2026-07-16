"""Analyzer integration for the exact preregistered Phase B kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rextio.analyzer.models import FunctionAnalysis, ProjectAnalysis
from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from benchmarks.phase_b_cases import KERNEL_SOURCE
from rextio_torch.claim.activations import RELU_RULE
from rextio_torch.claim.linear import LINEAR_RULE
from rextio_torch.claim.reductions import MEAN_RULE
from rextio_torch.plugin import PLUGIN_ID, plugin


class FakeEntryPoint:
    """Minimal installed-entry-point stand-in used by the core loader."""

    name = PLUGIN_ID

    def load(self) -> Any:
        """Return the in-tree plugin object."""

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


def test_exact_phase_b_kernel_routes_native_with_five_claims(tmp_path: Path) -> None:
    registry = load_plugin_registry(
        PluginConfig(enabled=(PLUGIN_ID,)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )
    analysis = analyze_project(
        _write_module(tmp_path),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )

    inference = _function(analysis, "myapp.kernels.inference")
    assert inference.accepted is True
    assert inference.route == f"native-plugin:{PLUGIN_ID}"
    assert not any(diagnostic.severity == "error" for diagnostic in inference.diagnostics)

    rule_ids = [claim.rule_id for claim in inference.plugin_claims]
    assert len(rule_ids) == 5
    assert rule_ids.count(LINEAR_RULE) == 2
    assert rule_ids.count(RELU_RULE) == 2
    assert rule_ids.count(MEAN_RULE) == 1
