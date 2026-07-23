"""Bounded function spellings and static dim/keepdim claim/lower contracts."""

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
    NotCovered,
    ReceiverMeta,
    Rejected,
)
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_torch.claim.activations import (
    FUNCTION_RELU_RULE,
    FUNCTION_SIGMOID_RULE,
    FUNCTION_TANH_RULE,
)
from rextio_torch.claim.classification import (
    ARGMAX_STATIC_RULE,
    SOFTMAX_STATIC_RULE,
)
from rextio_torch.claim.reductions import MEAN_STATIC_RULE, SUM_STATIC_RULE
from rextio_torch.diagnostics import (
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_torch.plugin import PLUGIN_ID, plugin
from rextio_torch.rust_snippets import (
    RELU,
    reduction_call_name,
    reduction_helper,
    softmax_call_name,
    softmax_helper,
)

PLUGIN = plugin()
CONFIG = RextioConfig()
NON_LITERAL = ClaimLiteral(is_literal=False, value=None)


def _kw(name: str, value: object) -> KeywordArg:
    return KeywordArg(
        name=name,
        arg_type=type(value).__name__,
        literal=ClaimLiteral(is_literal=True, value=value),
    )


def _call(
    target: str,
    input_type: str,
    *,
    positional_dim: object | None = None,
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    operands: tuple[str | None, ...] = (input_type,)
    literals: tuple[ClaimLiteral, ...] = (NON_LITERAL,)
    if positional_dim is not None:
        operands += ("int",)
        literals += (ClaimLiteral(is_literal=True, value=positional_dim),)
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operands,
        operand_literals=literals,
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
    )


def _method(
    method: str,
    input_type: str,
    *,
    positional_dim: object | None = None,
    keywords: tuple[KeywordArg, ...] = (),
    extra_positional_keepdim: bool | None = None,
) -> ClaimSite:
    operands: tuple[str | None, ...] = ()
    literals: tuple[ClaimLiteral, ...] = ()
    if positional_dim is not None:
        operands = ("int",)
        literals = (ClaimLiteral(is_literal=True, value=positional_dim),)
    if extra_positional_keepdim is not None:
        operands += ("bool",)
        literals += (
            ClaimLiteral(is_literal=True, value=extra_positional_keepdim),
        )
    return ClaimSite(
        kind="call",
        target=f"logits.{method}",
        operand_types=operands,
        operand_literals=literals,
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
        receiver=ReceiverMeta(
            arg_type=input_type,
            expr_kind="name",
            is_safe=True,
        ),
    )


def _fresh_name(prefix: str) -> str:
    return f"{prefix}_0"


@pytest.mark.parametrize(
    ("target", "rule_id"),
    (
        ("torch.relu", FUNCTION_RELU_RULE),
        ("torch.sigmoid", FUNCTION_SIGMOID_RULE),
        ("torch.tanh", FUNCTION_TANH_RULE),
    ),
)
@pytest.mark.parametrize("input_type", (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D))
def test_claims_exact_functional_activations(
    target: str,
    rule_id: str,
    input_type: str,
) -> None:
    assert PLUGIN.claim(_call(target, input_type), CONFIG) == Claimed(
        rule_id=rule_id,
        result_type=input_type,
    )


def test_functional_activation_canonical_name_and_arity_are_exact() -> None:
    noncanonical = _call("torch.nn.functional.relu", TENSOR_F32_CPU_2D)
    assert isinstance(PLUGIN.claim(noncanonical, CONFIG), NotCovered)
    extra = ClaimSite(
        kind="call",
        target="torch.relu",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
    )
    assert isinstance(PLUGIN.claim(extra, CONFIG), Rejected)


@pytest.mark.parametrize(
    ("site", "rule_id", "result_type"),
    (
        (
            _call("torch.mean", TENSOR_F32_CPU_2D, keywords=(_kw("dim", 0),)),
            MEAN_STATIC_RULE,
            TENSOR_F32_CPU_1D,
        ),
        (
            _call(
                "torch.sum",
                TENSOR_F32_CPU_2D,
                positional_dim=1,
                keywords=(_kw("keepdim", True),),
            ),
            SUM_STATIC_RULE,
            TENSOR_F32_CPU_2D,
        ),
        (
            _method(
                "mean",
                TENSOR_F32_CPU_1D,
                positional_dim=0,
                keywords=(_kw("keepdim", True),),
            ),
            MEAN_STATIC_RULE,
            TENSOR_F32_CPU_1D,
        ),
        (
            _method("sum", TENSOR_F32_CPU_2D, keywords=(_kw("dim", 1),)),
            SUM_STATIC_RULE,
            TENSOR_F32_CPU_1D,
        ),
    ),
)
def test_claims_representable_static_reductions(
    site: ClaimSite,
    rule_id: str,
    result_type: str,
) -> None:
    assert PLUGIN.claim(site, CONFIG) == Claimed(
        rule_id=rule_id,
        result_type=result_type,
    )


@pytest.mark.parametrize(
    "site",
    (
        _method("mean", TENSOR_F32_CPU_1D, positional_dim=0),
        _method(
            "sum",
            TENSOR_F32_CPU_2D,
            positional_dim=0,
            extra_positional_keepdim=True,
        ),
        _method(
            "mean",
            TENSOR_F32_CPU_2D,
            positional_dim=0,
            keywords=(_kw("dim", 1),),
        ),
    ),
)
def test_reduction_unrepresentable_or_ambiguous_forms_fallback(site: ClaimSite) -> None:
    assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)


def test_positional_dim_requires_aligned_literal_metadata() -> None:
    malformed = _method("mean", TENSOR_F32_CPU_2D, positional_dim=0)
    malformed = ClaimSite(
        kind=malformed.kind,
        target=malformed.target,
        operand_types=malformed.operand_types,
        operand_literals=(),
        file_path="",
        line=0,
        column=0,
        receiver=malformed.receiver,
    )
    assert isinstance(PLUGIN.claim(malformed, CONFIG), Rejected)


@pytest.mark.parametrize(
    ("site", "rule_id", "result_type"),
    (
        (
            _call("torch.softmax", TENSOR_F32_CPU_1D, positional_dim=0),
            SOFTMAX_STATIC_RULE,
            TENSOR_F32_CPU_1D,
        ),
        (
            _method("softmax", TENSOR_F32_CPU_2D, keywords=(_kw("dim", 0),)),
            SOFTMAX_STATIC_RULE,
            TENSOR_F32_CPU_2D,
        ),
        (
            _call("torch.argmax", TENSOR_F32_CPU_2D, keywords=(_kw("dim", 0),)),
            ARGMAX_STATIC_RULE,
            TENSOR_I64_CPU_1D,
        ),
        (
            _method(
                "argmax",
                TENSOR_F32_CPU_1D,
                positional_dim=0,
                keywords=(_kw("keepdim", True),),
            ),
            ARGMAX_STATIC_RULE,
            TENSOR_I64_CPU_1D,
        ),
    ),
)
def test_claims_representable_static_classification(
    site: ClaimSite,
    rule_id: str,
    result_type: str,
) -> None:
    assert PLUGIN.claim(site, CONFIG) == Claimed(
        rule_id=rule_id,
        result_type=result_type,
    )


@pytest.mark.parametrize(
    "site",
    (
        _method(
            "argmax",
            TENSOR_F32_CPU_2D,
            keywords=(_kw("dim", 1), _kw("keepdim", True)),
        ),
        _method("argmax", TENSOR_F32_CPU_1D, keywords=(_kw("dim", 0),)),
        _call(
            "torch.softmax",
            TENSOR_F32_CPU_2D,
            keywords=(_kw("dim", 1), _kw("keepdim", False)),
        ),
        _call(
            "torch.softmax",
            TENSOR_F32_CPU_2D,
            keywords=(_kw("dim", 1), _kw("dtype", "float32")),
        ),
    ),
)
def test_classification_unrepresentable_or_semantic_options_fallback(
    site: ClaimSite,
) -> None:
    assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)


def test_functional_activation_lower_uses_only_tensor_operand() -> None:
    claimed = _call("torch.relu", TENSOR_F32_CPU_2D)
    claimed = replace(
        claimed,
        rule_id=FUNCTION_RELU_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(
            operands=("tensor",),
            target_language="rust",
            fresh_name=_fresh_name,
        ),
    )
    assert lowered.rust == f"{RELU}(&tensor)?"


def test_positional_reduction_dim_is_static_not_runtime_tensor_input() -> None:
    claimed = _call(
        "torch.sum",
        TENSOR_F32_CPU_2D,
        positional_dim=0,
        keywords=(_kw("keepdim", True),),
    )
    claimed = replace(
        claimed,
        rule_id=SUM_STATIC_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(
            operands=("tensor", "rendered_dim"),
            target_language="rust",
            fresh_name=_fresh_name,
        ),
    )
    call_name = reduction_call_name("sum", 0, True)
    assert lowered.rust == f"{call_name}(&tensor)?"
    assert "rendered_dim" not in lowered.rust
    assert reduction_helper("sum", 0, True) in lowered.helpers


def test_positional_classification_dim_is_static_not_runtime_tensor_input() -> None:
    claimed = _call("torch.softmax", TENSOR_F32_CPU_1D, positional_dim=0)
    claimed = replace(
        claimed,
        rule_id=SOFTMAX_STATIC_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(
            operands=("tensor", "rendered_dim"),
            target_language="rust",
            fresh_name=_fresh_name,
        ),
    )
    assert lowered.rust == f"{softmax_call_name(0)}(&tensor)?"
    assert "rendered_dim" not in lowered.rust
    assert softmax_helper(0) in lowered.helpers


def test_lower_rejects_forged_positional_literal_alignment() -> None:
    claimed = _method("mean", TENSOR_F32_CPU_2D, positional_dim=0)
    claimed = ClaimSite(
        kind=claimed.kind,
        target=claimed.target,
        operand_types=claimed.operand_types,
        operand_literals=(),
        file_path="",
        line=0,
        column=0,
        rule_id=MEAN_STATIC_RULE,
        result_type=TENSOR_F32_CPU_1D,
        receiver=claimed.receiver,
    )
    with pytest.raises(ValueError, match="not aligned"):
        PLUGIN.lower(
            claimed,
            LoweringContext(
                operands=("rendered_dim",),
                target_language="rust",
                fresh_name=_fresh_name,
                receiver="tensor",
            ),
        )


@pytest.mark.parametrize(
    ("site", "rule_id", "result_type"),
    (
        (
            _method(
                "mean",
                TENSOR_F32_CPU_2D,
                keywords=(replace(_kw("dim", 0), arg_type="bool"),),
            ),
            MEAN_STATIC_RULE,
            TENSOR_F32_CPU_1D,
        ),
        (
            _method(
                "sum",
                TENSOR_F32_CPU_2D,
                keywords=(
                    _kw("dim", 0),
                    replace(_kw("keepdim", True), arg_type="int"),
                ),
            ),
            SUM_STATIC_RULE,
            TENSOR_F32_CPU_2D,
        ),
    ),
)
def test_reduction_keyword_arg_type_must_match_literal_kind(
    site: ClaimSite,
    rule_id: str,
    result_type: str,
) -> None:
    assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)
    forged = replace(site, rule_id=rule_id, result_type=result_type)
    with pytest.raises(ValueError, match="literal keywords"):
        PLUGIN.lower(
            forged,
            LoweringContext(
                operands=(),
                target_language="rust",
                fresh_name=_fresh_name,
                receiver="tensor",
            ),
        )


@pytest.mark.parametrize(
    ("site", "rule_id", "result_type"),
    (
        (
            _method(
                "softmax",
                TENSOR_F32_CPU_2D,
                keywords=(replace(_kw("dim", 0), arg_type="bool"),),
            ),
            SOFTMAX_STATIC_RULE,
            TENSOR_F32_CPU_2D,
        ),
        (
            _method(
                "argmax",
                TENSOR_F32_CPU_1D,
                keywords=(
                    _kw("dim", 0),
                    replace(_kw("keepdim", True), arg_type="int"),
                ),
            ),
            ARGMAX_STATIC_RULE,
            TENSOR_I64_CPU_1D,
        ),
    ),
)
def test_classification_keyword_arg_type_must_match_literal_kind(
    site: ClaimSite,
    rule_id: str,
    result_type: str,
) -> None:
    assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)
    forged = replace(site, rule_id=rule_id, result_type=result_type)
    with pytest.raises(ValueError, match="literal keywords"):
        PLUGIN.lower(
            forged,
            LoweringContext(
                operands=(),
                target_language="rust",
                fresh_name=_fresh_name,
                receiver="tensor",
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


def _function(analysis: ProjectAnalysis, qualname: str) -> FunctionAnalysis:
    for module in analysis.modules:
        for function in module.functions:
            if function.qualname == qualname:
                return function
    raise AssertionError(f"function {qualname!r} not found")


FUNCTION_SOURCE = """
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D, TensorI64Cpu1D
import torch


def function_surface(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    activated = torch.relu(x)
    reduced = torch.sum(activated, 0, keepdim=True)
    probabilities = torch.softmax(reduced, dim=1)
    return torch.tanh(probabilities)


def function_argmax(x: TensorF32Cpu2D) -> TensorI64Cpu1D:
    probabilities = torch.sigmoid(x)
    return torch.argmax(probabilities, dim=0)


def vector_argmax(x: TensorF32Cpu1D) -> TensorI64Cpu1D:
    return torch.argmax(x, 0, keepdim=True)


def invalid_positional_keepdim(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return torch.mean(x, 0, True)


def invalid_rank2_keepdim_argmax(x: TensorF32Cpu2D) -> TensorI64Cpu1D:
    return torch.argmax(x, dim=1, keepdim=True)


def invalid_rounded_div(
    left: TensorF32Cpu2D,
    right: TensorF32Cpu2D,
) -> TensorF32Cpu2D:
    return torch.div(left, right, rounding_mode="trunc")
"""


def test_analyzer_routes_new_exact_function_spellings(tmp_path: Path) -> None:
    registry = _registry()
    analysis = analyze_project(
        _write_module(tmp_path, FUNCTION_SOURCE),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    expected = {
        "function_surface": [
            FUNCTION_RELU_RULE,
            SUM_STATIC_RULE,
            SOFTMAX_STATIC_RULE,
            FUNCTION_TANH_RULE,
        ],
        "function_argmax": [FUNCTION_SIGMOID_RULE, ARGMAX_STATIC_RULE],
        "vector_argmax": [ARGMAX_STATIC_RULE],
    }
    for name, rules in expected.items():
        function = _function(analysis, f"myapp.kernels.{name}")
        assert function.accepted is True
        assert function.route == f"native-plugin:{PLUGIN_ID}"
        assert [claim.rule_id for claim in function.plugin_claims] == rules
        assert not any(d.severity == "error" for d in function.diagnostics)
    for name in (
        "invalid_positional_keepdim",
        "invalid_rank2_keepdim_argmax",
        "invalid_rounded_div",
    ):
        function = _function(analysis, f"myapp.kernels.{name}")
        assert function.route != f"native-plugin:{PLUGIN_ID}"
        assert function.plugin_claims == []


def test_static_rule_records_are_native_verified() -> None:
    by_id = {record.id: record for record in PLUGIN.describe(CONFIG)}
    for rule_id in (
        FUNCTION_RELU_RULE,
        FUNCTION_SIGMOID_RULE,
        FUNCTION_TANH_RULE,
        MEAN_STATIC_RULE,
        SUM_STATIC_RULE,
        SOFTMAX_STATIC_RULE,
        ARGMAX_STATIC_RULE,
    ):
        assert by_id[rule_id].outcome == "native"
        assert by_id[rule_id].verified is True


def test_functional_div_with_rounding_mode_is_not_claimed() -> None:
    site = ClaimSite(
        kind="call",
        target="torch.div",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        keywords=(_kw("rounding_mode", "trunc"),),
    )
    assert isinstance(PLUGIN.claim(site, CONFIG), NotCovered)
