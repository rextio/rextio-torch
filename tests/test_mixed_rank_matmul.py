"""Mixed-rank matrix/vector matmul across all exact supported spellings."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rextio.config.schema import RextioConfig
from rextio.plugins.api import Claimed, ClaimSite, LoweringContext, ReceiverMeta, Rejected

from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.plugin import plugin
from rextio_torch.rust_snippets import MATMUL, matmul_helper

PLUGIN = plugin()
CONFIG = RextioConfig()
BINOP_RULE = "rextio-torch/tensor-matmul-f32-cpu-mixed-rank"
CALL_RULE = "rextio-torch/tensor-matmul-call-f32-cpu-mixed-rank"
MIXED_PAIRS = (
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D),
)


def _binop(left: str, right: str) -> ClaimSite:
    return ClaimSite(
        kind="binop",
        target="@",
        operand_types=(left, right),
        file_path="",
        line=0,
        column=0,
    )


def _function(left: str, right: str) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target="torch.matmul",
        operand_types=(left, right),
        file_path="",
        line=0,
        column=0,
    )


def _method(receiver_type: str, other_type: str) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target="torch.Tensor.matmul",
        operand_types=(other_type,),
        file_path="",
        line=0,
        column=0,
        receiver=ReceiverMeta(
            arg_type=receiver_type,
            expr_kind="name",
            is_safe=True,
        ),
    )


@pytest.mark.parametrize(("left", "right"), MIXED_PAIRS)
def test_claims_mixed_rank_matmul_for_every_exact_spelling(left: str, right: str) -> None:
    assert PLUGIN.claim(_binop(left, right), CONFIG) == Claimed(
        rule_id=BINOP_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    assert PLUGIN.claim(_function(left, right), CONFIG) == Claimed(
        rule_id=CALL_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    assert PLUGIN.claim(_method(left, right), CONFIG) == Claimed(
        rule_id=CALL_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )


def test_rank1_rank1_matmul_remains_rejected_because_rank0_is_unregistered() -> None:
    for site in (
        _binop(TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
        _function(TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
        _method(TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
    ):
        assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)


@pytest.mark.parametrize(("left", "right"), MIXED_PAIRS)
@pytest.mark.parametrize(("form", "rule_id"), (("binop", BINOP_RULE), ("function", CALL_RULE)))
def test_lowers_mixed_rank_matmul_binop_and_function(
    left: str,
    right: str,
    form: str,
    rule_id: str,
) -> None:
    site = _binop(left, right) if form == "binop" else _function(left, right)
    claimed = replace(site, rule_id=rule_id, result_type=TENSOR_F32_CPU_1D)

    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(
            operands=("left", "right"),
            target_language="rust",
            fresh_name=lambda prefix: f"{prefix}_0",
        ),
    )

    assert lowered.rust == f"{MATMUL}(&left, &right)?"
    assert matmul_helper() in lowered.helpers
    assert "f_matmul" in matmul_helper()
    assert "no_grad_guard" in matmul_helper()


@pytest.mark.parametrize(("receiver_type", "other_type"), MIXED_PAIRS)
def test_lowers_mixed_rank_matmul_method(receiver_type: str, other_type: str) -> None:
    claimed = replace(
        _method(receiver_type, other_type),
        rule_id=CALL_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )

    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(
            operands=("other",),
            receiver="receiver",
            target_language="rust",
            fresh_name=lambda prefix: f"{prefix}_0",
        ),
    )

    assert lowered.rust == f"{MATMUL}(&receiver, &other)?"


def test_lower_revalidates_mixed_rank_matmul_metadata() -> None:
    valid_binop = replace(
        _binop(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
        rule_id=BINOP_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    valid_function = replace(
        _function(TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D),
        rule_id=CALL_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    valid_method = replace(
        _method(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
        rule_id=CALL_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    no_receiver = LoweringContext(
        operands=("left", "right"),
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )
    forged_binops = (
        replace(valid_binop, rule_id=CALL_RULE),
        replace(valid_binop, result_type=TENSOR_F32_CPU_2D),
        replace(
            valid_binop,
            operand_types=(TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
        ),
        replace(
            valid_binop,
            operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        ),
    )
    for forged in forged_binops:
        with pytest.raises(ValueError):
            PLUGIN.lower(forged, no_receiver)

    forged_function = replace(
        valid_function,
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CPU_1D,
            expr_kind="name",
            is_safe=True,
        ),
    )
    with pytest.raises(ValueError):
        PLUGIN.lower(forged_function, no_receiver)

    method_context = LoweringContext(
        operands=("other",),
        receiver="receiver",
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )
    for forged in (
        replace(valid_method, rule_id=BINOP_RULE),
        replace(
            valid_method,
            receiver=ReceiverMeta(
                arg_type=TENSOR_F32_CPU_1D,
                expr_kind="name",
                is_safe=True,
            ),
        ),
    ):
        with pytest.raises(ValueError):
            PLUGIN.lower(forged, method_context)

    for site, context in (
        (valid_binop, replace(no_receiver, operands=("left",))),
        (valid_function, replace(no_receiver, operands=("left", "right", "extra"))),
        (valid_method, replace(method_context, receiver=None)),
        (valid_method, replace(method_context, operands=())),
    ):
        with pytest.raises(ValueError):
            PLUGIN.lower(site, context)
