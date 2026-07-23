"""Focused claim/lower coverage for bounded tensor subtraction and division."""

from __future__ import annotations

import pytest
from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimSite, LoweringContext, Rejected

from rextio_torch.claim.binops import (
    DIV_BROADCAST_2D_1D_RULE,
    DIV_SAME_RANK_RULE,
    SUB_BROADCAST_2D_1D_RULE,
    SUB_SAME_RANK_RULE,
)
from rextio_torch.diagnostics import (
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_torch.plugin import plugin
from rextio_torch.rust_snippets import DIV, SUB, div_helper, sub_helper

PLUGIN = plugin()
CONFIG = RextioConfig()


def _site(op: str, left: str | None, right: str | None) -> ClaimSite:
    return ClaimSite(
        kind="binop",
        target=op,
        operand_types=(left, right),
        file_path="",
        line=0,
        column=0,
    )


def _fresh_name(prefix: str) -> str:
    return f"{prefix}_0"


@pytest.mark.parametrize(
    ("op", "left", "right", "rule_id", "result_type"),
    (
        ("-", TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D, SUB_SAME_RANK_RULE, TENSOR_F32_CPU_1D),
        ("-", TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, SUB_SAME_RANK_RULE, TENSOR_F32_CPU_2D),
        (
            "-",
            TENSOR_F32_CPU_2D,
            TENSOR_F32_CPU_1D,
            SUB_BROADCAST_2D_1D_RULE,
            TENSOR_F32_CPU_2D,
        ),
        (
            "-",
            TENSOR_F32_CPU_1D,
            TENSOR_F32_CPU_2D,
            SUB_BROADCAST_2D_1D_RULE,
            TENSOR_F32_CPU_2D,
        ),
        ("/", TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D, DIV_SAME_RANK_RULE, TENSOR_F32_CPU_1D),
        ("/", TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, DIV_SAME_RANK_RULE, TENSOR_F32_CPU_2D),
        (
            "/",
            TENSOR_F32_CPU_2D,
            TENSOR_F32_CPU_1D,
            DIV_BROADCAST_2D_1D_RULE,
            TENSOR_F32_CPU_2D,
        ),
        (
            "/",
            TENSOR_F32_CPU_1D,
            TENSOR_F32_CPU_2D,
            DIV_BROADCAST_2D_1D_RULE,
            TENSOR_F32_CPU_2D,
        ),
    ),
)
def test_claims_bounded_tensor_subtraction_and_division(
    op: str,
    left: str,
    right: str,
    rule_id: str,
    result_type: str,
) -> None:
    assert PLUGIN.claim(_site(op, left, right), CONFIG) == Claimed(
        rule_id=rule_id,
        result_type=result_type,
    )


@pytest.mark.parametrize("op", ("-", "/"))
def test_rejects_scalar_and_classification_operands(op: str) -> None:
    for right in ("int", TENSOR_I64_CPU_1D):
        result = PLUGIN.claim(_site(op, TENSOR_F32_CPU_1D, right), CONFIG)
        assert isinstance(result, Rejected)
        assert result.diagnostic.code == DIAGNOSTIC_UNSUPPORTED


@pytest.mark.parametrize(
    ("op", "rule_id", "left", "right", "result_type", "call_name", "helper", "tch_call"),
    (
        (
            "-",
            SUB_SAME_RANK_RULE,
            TENSOR_F32_CPU_1D,
            TENSOR_F32_CPU_1D,
            TENSOR_F32_CPU_1D,
            SUB,
            sub_helper(),
            "f_sub",
        ),
        (
            "-",
            SUB_BROADCAST_2D_1D_RULE,
            TENSOR_F32_CPU_1D,
            TENSOR_F32_CPU_2D,
            TENSOR_F32_CPU_2D,
            SUB,
            sub_helper(),
            "f_sub",
        ),
        (
            "/",
            DIV_SAME_RANK_RULE,
            TENSOR_F32_CPU_2D,
            TENSOR_F32_CPU_2D,
            TENSOR_F32_CPU_2D,
            DIV,
            div_helper(),
            "f_div",
        ),
        (
            "/",
            DIV_BROADCAST_2D_1D_RULE,
            TENSOR_F32_CPU_2D,
            TENSOR_F32_CPU_1D,
            TENSOR_F32_CPU_2D,
            DIV,
            div_helper(),
            "f_div",
        ),
    ),
)
def test_lowers_bounded_tensor_subtraction_and_division(
    op: str,
    rule_id: str,
    left: str,
    right: str,
    result_type: str,
    call_name: str,
    helper: str,
    tch_call: str,
) -> None:
    claimed = ClaimSite(
        kind="binop",
        target=op,
        operand_types=(left, right),
        file_path="",
        line=0,
        column=0,
        rule_id=rule_id,
        result_type=result_type,
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(
            operands=("left", "right"),
            target_language="rust",
            fresh_name=_fresh_name,
        ),
    )
    assert lowered.rust == f"{call_name}(&left, &right)?"
    assert helper in lowered.helpers
    assert tch_call in helper
    assert "no_grad_guard" in helper
    assert "unwrap(" not in helper


@pytest.mark.parametrize(
    ("op", "rule_id"),
    (("-", SUB_SAME_RANK_RULE), ("/", DIV_SAME_RANK_RULE)),
)
def test_lower_rejects_forged_non_float_metadata(op: str, rule_id: str) -> None:
    claimed = ClaimSite(
        kind="binop",
        target=op,
        operand_types=(TENSOR_I64_CPU_1D, TENSOR_I64_CPU_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=rule_id,
        result_type=TENSOR_I64_CPU_1D,
    )
    with pytest.raises(ValueError, match="float32 CPU"):
        PLUGIN.lower(
            claimed,
            LoweringContext(
                operands=("left", "right"),
                target_language="rust",
                fresh_name=_fresh_name,
            ),
        )


@pytest.mark.parametrize(
    ("op", "rule_id"),
    (("-", SUB_SAME_RANK_RULE), ("/", DIV_SAME_RANK_RULE)),
)
def test_lower_rejects_forged_receiver(op: str, rule_id: str) -> None:
    claimed = ClaimSite(
        kind="binop",
        target=op,
        operand_types=(TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=rule_id,
        result_type=TENSOR_F32_CPU_1D,
    )
    with pytest.raises(ValueError, match="two positional"):
        PLUGIN.lower(
            claimed,
            LoweringContext(
                operands=("left", "right"),
                target_language="rust",
                fresh_name=_fresh_name,
                receiver="forged",
            ),
        )
