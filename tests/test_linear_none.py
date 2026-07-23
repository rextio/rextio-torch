"""Prove-before-claim coverage for bounded functional linear without bias."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from rextio.analyzer.models import FunctionAnalysis, ProjectAnalysis
from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.api import (
    ClaimLiteral,
    Claimed,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    Rejected,
)
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_torch.claim.linear import (
    LINEAR_NO_BIAS_RULE,
    LINEAR_TARGET,
)
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.plugin import PLUGIN_ID, plugin
from rextio_torch.rust_snippets import (
    LINEAR_NO_BIAS,
    linear_no_bias_helper,
)

PLUGIN = plugin()
CONFIG = RextioConfig()
NON_LITERAL = ClaimLiteral(is_literal=False, value=None)
LITERAL_NONE = ClaimLiteral(is_literal=True, value=None)


def _site(form: str) -> ClaimSite:
    if form == "omitted":
        return ClaimSite(
            kind="call",
            target=LINEAR_TARGET,
            operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
            operand_literals=(NON_LITERAL, NON_LITERAL),
            file_path="",
            line=0,
            column=0,
        )
    if form == "positional":
        return ClaimSite(
            kind="call",
            target=LINEAR_TARGET,
            operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, "None"),
            operand_literals=(NON_LITERAL, NON_LITERAL, LITERAL_NONE),
            file_path="",
            line=0,
            column=0,
        )
    if form == "keyword":
        return ClaimSite(
            kind="call",
            target=LINEAR_TARGET,
            operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
            operand_literals=(NON_LITERAL, NON_LITERAL),
            file_path="",
            line=0,
            column=0,
            keywords=(
                KeywordArg(
                    name="bias",
                    arg_type="None",
                    literal=LITERAL_NONE,
                ),
            ),
        )
    raise AssertionError(form)


def _fresh_name(prefix: str) -> str:
    return f"{prefix}_0"


@pytest.mark.parametrize("form", ("omitted", "positional", "keyword"))
def test_claims_proved_none_bias_forms(form: str) -> None:
    assert PLUGIN.claim(_site(form), CONFIG) == Claimed(
        rule_id=LINEAR_NO_BIAS_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )


@pytest.mark.parametrize("form", ("omitted", "positional", "keyword"))
def test_lowers_proved_none_bias_forms_without_runtime_bias(form: str) -> None:
    site = replace(
        _site(form),
        rule_id=LINEAR_NO_BIAS_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    operands = ("input", "weight", "rendered_none") if form == "positional" else (
        "input",
        "weight",
    )
    lowered = PLUGIN.lower(
        site,
        LoweringContext(
            operands=operands,
            target_language="rust",
            fresh_name=_fresh_name,
        ),
    )
    assert lowered.rust == f"{LINEAR_NO_BIAS}(&input, &weight)?"
    assert "rendered_none" not in lowered.rust
    assert linear_no_bias_helper() in lowered.helpers
    assert "Option::<&tch::Tensor>::None" in linear_no_bias_helper()
    assert "no_grad_guard" in linear_no_bias_helper()


def test_positional_none_requires_aligned_literal_metadata_at_claim_and_lower() -> None:
    malformed = replace(_site("positional"), operand_literals=())
    assert isinstance(PLUGIN.claim(malformed, CONFIG), Rejected)
    forged = replace(
        malformed,
        rule_id=LINEAR_NO_BIAS_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    with pytest.raises(ValueError, match="literal-None"):
        PLUGIN.lower(
            forged,
            LoweringContext(
                operands=("input", "weight", "rendered_none"),
                target_language="rust",
                fresh_name=_fresh_name,
            ),
        )


def test_tensor_valued_keyword_bias_is_not_misclaimed() -> None:
    forged_keyword_site = ClaimSite(
        kind="call",
        target=LINEAR_TARGET,
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        keywords=(
            KeywordArg(
                name="bias",
                arg_type=TENSOR_F32_CPU_1D,
                literal=NON_LITERAL,
            ),
        ),
    )
    assert isinstance(PLUGIN.claim(forged_keyword_site, CONFIG), Rejected)


@pytest.mark.parametrize("form", ("positional", "keyword"))
def test_contradictory_tensor_type_plus_none_literal_is_rejected(form: str) -> None:
    site = _site(form)
    if form == "positional":
        contradictory = replace(
            site,
            operand_types=(
                TENSOR_F32_CPU_2D,
                TENSOR_F32_CPU_2D,
                TENSOR_F32_CPU_1D,
            ),
        )
    else:
        contradictory = replace(
            site,
            keywords=(
                KeywordArg(
                    name="bias",
                    arg_type=TENSOR_F32_CPU_1D,
                    literal=LITERAL_NONE,
                ),
            ),
        )
    assert isinstance(PLUGIN.claim(contradictory, CONFIG), Rejected)

    forged = replace(
        contradictory,
        rule_id=LINEAR_NO_BIAS_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    operands = (
        ("input", "weight", "forged_tensor")
        if form == "positional"
        else ("input", "weight")
    )
    with pytest.raises(ValueError, match="literal-None"):
        PLUGIN.lower(
            forged,
            LoweringContext(
                operands=operands,
                target_language="rust",
                fresh_name=_fresh_name,
            ),
        )


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


def _function(analysis: ProjectAnalysis, name: str) -> FunctionAnalysis:
    for module in analysis.modules:
        for function in module.functions:
            if function.qualname == f"myapp.kernels.{name}":
                return function
    raise AssertionError(name)


LINEAR_SOURCE = """
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch.nn.functional as F


def omitted(x: TensorF32Cpu2D, weight: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return F.linear(x, weight)


def positional_none(x: TensorF32Cpu2D, weight: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return F.linear(x, weight, None)


def keyword_none(x: TensorF32Cpu2D, weight: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return F.linear(x, weight, bias=None)


def tensor_keyword(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu2D:
    return F.linear(x, weight, bias=bias)
"""


def test_analyzer_proves_all_none_forms_and_falls_back_for_tensor_keyword(
    tmp_path: Path,
) -> None:
    registry = _registry()
    analysis = analyze_project(
        _write_module(tmp_path, LINEAR_SOURCE),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    for name in ("omitted", "positional_none", "keyword_none"):
        function = _function(analysis, name)
        assert function.accepted is True
        assert function.route == f"native-plugin:{PLUGIN_ID}"
        assert [claim.rule_id for claim in function.plugin_claims] == [
            LINEAR_NO_BIAS_RULE
        ]
    tensor_keyword = _function(analysis, "tensor_keyword")
    assert tensor_keyword.accepted is False
    assert tensor_keyword.route != f"native-plugin:{PLUGIN_ID}"
    assert tensor_keyword.plugin_claims == []
    # Core API 1.3 cannot encode a runtime tensor keyword operand in ClaimSite,
    # so the call is never offered to this provider and follows ordinary fallback.


def test_linear_none_rule_record_is_native_verified() -> None:
    record = next(
        record
        for record in PLUGIN.describe(CONFIG)
        if record.id == LINEAR_NO_BIAS_RULE
    )
    assert record.outcome == "native"
    assert record.verified is True
