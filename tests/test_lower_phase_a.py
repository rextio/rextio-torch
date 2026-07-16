"""Focused lower/codegen tests for the Phase A slice (no real Cargo)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from rextio.plugins.api import (
    ClaimLiteral,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    ReceiverMeta,
)

from rextio_torch.claim.activations import RELU_RULE
from rextio_torch.claim.linear import LINEAR_RULE, LINEAR_TARGET
from rextio_torch.claim.reductions import MEAN_RULE
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.plugin import plugin
from rextio_torch.rust_snippets import (
    LINEAR,
    MEAN_DIM1_KEEPFALSE,
    RELU,
    boundary_helpers,
    linear_helper,
    mean_dim1_keepfalse_helper,
    relu_helper,
)

PLUGIN = plugin()


def _fresh_name(prefix: str) -> str:
    return f"{prefix}_0"


def test_lower_linear_emits_fallible_no_grad_helper() -> None:
    claimed = ClaimSite(
        kind="call",
        target=LINEAR_TARGET,
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=LINEAR_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    ctx = LoweringContext(
        operands=("x", "w", "b"),
        target_language="rust",
        fresh_name=_fresh_name,
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == f"{LINEAR}(&x, &w, &b)?"
    assert boundary_helpers() in lowered.helpers
    assert linear_helper() in lowered.helpers
    assert "f_linear" in linear_helper()
    assert "no_grad_guard" in linear_helper()
    assert "unwrap(" not in linear_helper()
    assert "panic!" not in linear_helper()


def test_lower_relu_uses_receiver() -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensor.relu",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        rule_id=RELU_RULE,
        result_type=TENSOR_F32_CPU_2D,
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CPU_2D, expr_kind="call", is_safe=False
        ),
    )
    ctx = LoweringContext(
        operands=(),
        target_language="rust",
        fresh_name=_fresh_name,
        receiver="tmp_linear",
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == f"{RELU}(&tmp_linear)?"
    assert relu_helper() in lowered.helpers
    assert "f_relu" in relu_helper()
    assert "f_relu_" not in relu_helper()  # not in-place


def test_lower_mean_dim1_keepdim_false() -> None:
    keywords = (
        KeywordArg(name="dim", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
        KeywordArg(
            name="keepdim",
            arg_type="bool",
            literal=ClaimLiteral(is_literal=True, value=False),
        ),
    )
    claimed = ClaimSite(
        kind="call",
        target="mean",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        rule_id=MEAN_RULE,
        result_type=TENSOR_F32_CPU_1D,
        keywords=keywords,
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CPU_2D, expr_kind="call", is_safe=False
        ),
    )
    ctx = LoweringContext(
        operands=(),
        target_language="rust",
        fresh_name=_fresh_name,
        receiver="tmp_relu",
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == f"{MEAN_DIM1_KEEPFALSE}(&tmp_relu)?"
    assert mean_dim1_keepfalse_helper() in lowered.helpers
    assert "f_mean_dim" in mean_dim1_keepfalse_helper()
    assert "no_grad_guard" in mean_dim1_keepfalse_helper()


def test_lower_revalidates_mean_keywords() -> None:
    keywords = (
        KeywordArg(name="dim", arg_type="int", literal=ClaimLiteral(is_literal=True, value=0)),
        KeywordArg(
            name="keepdim",
            arg_type="bool",
            literal=ClaimLiteral(is_literal=True, value=False),
        ),
    )
    claimed = ClaimSite(
        kind="call",
        target="tensor.mean",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        rule_id=MEAN_RULE,
        result_type=TENSOR_F32_CPU_1D,
        keywords=keywords,
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CPU_2D, expr_kind="name", is_safe=True
        ),
    )
    ctx = LoweringContext(
        operands=(),
        target_language="rust",
        fresh_name=_fresh_name,
        receiver="t",
    )
    with pytest.raises(ValueError, match="dim=1"):
        PLUGIN.lower(claimed, ctx)


def test_lower_rejects_malformed_linear_metadata() -> None:
    claimed = ClaimSite(
        kind="call",
        target=LINEAR_TARGET,
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=LINEAR_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    ctx = LoweringContext(
        operands=("x", "w"),
        target_language="rust",
        fresh_name=_fresh_name,
    )
    with pytest.raises(ValueError, match="three positional"):
        PLUGIN.lower(claimed, ctx)


def test_lower_guard_survives_optimized_interpreter() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    program = (
        "from rextio.plugins.api import ClaimSite, LoweringContext\n"
        "from rextio_torch.claim.linear import LINEAR_RULE, LINEAR_TARGET\n"
        "from rextio_torch.diagnostics import TENSOR_F32_CPU_2D\n"
        "from rextio_torch.plugin import plugin\n"
        "claimed = ClaimSite(\n"
        "    kind='call', target=LINEAR_TARGET,\n"
        "    operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),\n"
        "    file_path='', line=0, column=0,\n"
        "    rule_id=LINEAR_RULE, result_type=TENSOR_F32_CPU_2D,\n"
        ")\n"
        "ctx = LoweringContext(operands=('a','b'), target_language='rust',\n"
        "                     fresh_name=lambda p: p)\n"
        "try:\n"
        "    plugin().lower(claimed, ctx)\n"
        "except ValueError as exc:\n"
        "    print('guarded', 'three' in str(exc).lower() or 'operand' in str(exc).lower())\n"
        "else:\n"
        "    print('missing-guard')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(src), env.get("PYTHONPATH", "")])
    completed = subprocess.run(
        [sys.executable, "-O", "-c", program],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "guarded True" in completed.stdout


def test_ops_helpers_map_errors_without_panic() -> None:
    for helper in (linear_helper(), relu_helper(), mean_dim1_keepfalse_helper()):
        assert "map_err(__rxttorch_map_err)" in helper
        assert "unwrap(" not in helper
        assert "expect(" not in helper
        assert "panic!" not in helper
