"""Focused lower/codegen tests for the Alpha AOT surface (no real Cargo)."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from typing import cast

import pytest
from rextio.plugins.api import (
    ClaimLiteral,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    ReceiverMeta,
)

from rextio_torch.claim.activations import RELU_RULE, SIGMOID_RULE_2D, TANH_RULE_2D
from rextio_torch.claim.binops import (
    ADD_BROADCAST_2D_1D_RULE,
    ADD_SAME_RANK_RULE,
    MATMUL_BINOP_RULE,
    MATMUL_CALL_RULE,
    MATMUL_CALL_TARGET,
)
from rextio_torch.claim.linear import LINEAR_RULE, LINEAR_TARGET
from rextio_torch.claim.reductions import MEAN_RULE, SUM_RULE
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.plugin import plugin
from rextio_torch.rust_snippets import (
    ADD,
    LINEAR,
    MATMUL,
    MEAN_DIM1_KEEPFALSE,
    RELU,
    SIGMOID,
    SUM_DIM1_KEEPFALSE,
    TANH,
    add_helper,
    boundary_helpers,
    linear_helper,
    matmul_helper,
    mean_dim1_keepfalse_helper,
    relu_helper,
    sigmoid_helper,
    sum_dim1_keepfalse_helper,
    tanh_helper,
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


def test_lower_defaults_legacy_contexts_to_pyo3_and_rejects_standalone() -> None:
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
    legacy_ctx = cast(
        LoweringContext,
        SimpleNamespace(operands=("x", "w", "b"), fresh_name=_fresh_name),
    )
    assert PLUGIN.lower(claimed, legacy_ctx).rust == f"{LINEAR}(&x, &w, &b)?"

    standalone_ctx = cast(
        LoweringContext,
        SimpleNamespace(
            backend="standalone-rust",
            operands=("x", "w", "b"),
            fresh_name=_fresh_name,
        ),
    )
    with pytest.raises(ValueError, match="PyO3"):
        PLUGIN.lower(claimed, standalone_ctx)


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


def test_lower_sigmoid_and_tanh() -> None:
    for rule, method, call_name, helper in (
        (SIGMOID_RULE_2D, "sigmoid", SIGMOID, sigmoid_helper()),
        (TANH_RULE_2D, "tanh", TANH, tanh_helper()),
    ):
        claimed = ClaimSite(
            kind="call",
            target=method,
            operand_types=(),
            file_path="",
            line=0,
            column=0,
            rule_id=rule,
            result_type=TENSOR_F32_CPU_2D,
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
        lowered = PLUGIN.lower(claimed, ctx)
        assert lowered.rust == f"{call_name}(&t)?"
        assert helper in lowered.helpers


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


def test_lower_sum_dim1_keepdim_false() -> None:
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
        target="tensor.sum",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        rule_id=SUM_RULE,
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
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == f"{SUM_DIM1_KEEPFALSE}(&t)?"
    assert sum_dim1_keepfalse_helper() in lowered.helpers
    assert "f_sum_dim_intlist" in sum_dim1_keepfalse_helper()


def test_lower_add_and_matmul() -> None:
    add_claimed = ClaimSite(
        kind="binop",
        target="+",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=ADD_BROADCAST_2D_1D_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    add_lowered = PLUGIN.lower(
        add_claimed,
        LoweringContext(operands=("x", "b"), target_language="rust", fresh_name=_fresh_name),
    )
    assert add_lowered.rust == f"{ADD}(&x, &b)?"
    assert add_helper() in add_lowered.helpers

    same = ClaimSite(
        kind="binop",
        target="+",
        operand_types=(TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=ADD_SAME_RANK_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    same_lowered = PLUGIN.lower(
        same,
        LoweringContext(operands=("a", "b"), target_language="rust", fresh_name=_fresh_name),
    )
    assert same_lowered.rust == f"{ADD}(&a, &b)?"

    mat = ClaimSite(
        kind="binop",
        target="@",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_BINOP_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    mat_lowered = PLUGIN.lower(
        mat,
        LoweringContext(operands=("a", "w"), target_language="rust", fresh_name=_fresh_name),
    )
    assert mat_lowered.rust == f"{MATMUL}(&a, &w)?"
    assert matmul_helper() in mat_lowered.helpers
    assert "f_matmul" in matmul_helper()


def test_lower_matmul_call_forms_revalidate_target_receiver_and_operands() -> None:
    functional = ClaimSite(
        kind="call",
        target=MATMUL_CALL_TARGET,
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_CALL_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    lowered = PLUGIN.lower(
        functional,
        LoweringContext(
            operands=("left", "right"),
            target_language="rust",
            fresh_name=_fresh_name,
        ),
    )
    assert lowered.rust == f"{MATMUL}(&left, &right)?"

    method = ClaimSite(
        kind="call",
        target="tensor.matmul",
        operand_types=(TENSOR_F32_CPU_2D,),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_CALL_RULE,
        result_type=TENSOR_F32_CPU_2D,
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CPU_2D,
            expr_kind="name",
            is_safe=True,
        ),
    )
    lowered = PLUGIN.lower(
        method,
        LoweringContext(
            operands=("right",),
            target_language="rust",
            fresh_name=_fresh_name,
            receiver="left",
        ),
    )
    assert lowered.rust == f"{MATMUL}(&left, &right)?"

    malformed_functional = ClaimSite(
        kind="call",
        target=MATMUL_CALL_TARGET,
        operand_types=(TENSOR_F32_CPU_2D,),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_CALL_RULE,
        result_type=TENSOR_F32_CPU_2D,
        receiver=method.receiver,
    )
    with pytest.raises(ValueError, match="functional torch.matmul"):
        PLUGIN.lower(
            malformed_functional,
            LoweringContext(
                operands=("right",),
                target_language="rust",
                fresh_name=_fresh_name,
                receiver="left",
            ),
        )

    malformed_method = ClaimSite(
        kind="call",
        target="tensor.matmul",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_CALL_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    with pytest.raises(ValueError, match="method .matmul"):
        PLUGIN.lower(
            malformed_method,
            LoweringContext(
                operands=("left", "right"),
                target_language="rust",
                fresh_name=_fresh_name,
            ),
        )

    wrong_rule = ClaimSite(
        kind="call",
        target=MATMUL_CALL_TARGET,
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_BINOP_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    with pytest.raises(ValueError, match="mismatched rule_id"):
        PLUGIN.lower(
            wrong_rule,
            LoweringContext(
                operands=("left", "right"),
                target_language="rust",
                fresh_name=_fresh_name,
            ),
        )


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
    helpers = (
        linear_helper(),
        relu_helper(),
        sigmoid_helper(),
        tanh_helper(),
        mean_dim1_keepfalse_helper(),
        sum_dim1_keepfalse_helper(),
        add_helper(),
        matmul_helper(),
    )
    for helper in helpers:
        assert "map_err(__rxttorch_map_err)" in helper
        assert "unwrap(" not in helper
        assert "expect(" not in helper
        assert "panic!" not in helper
        assert "no_grad_guard" in helper
