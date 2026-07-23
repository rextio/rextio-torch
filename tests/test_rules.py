"""Rule surface integrity for Alpha AOT."""

from __future__ import annotations

from rextio_torch.claim.activations import (
    RELU_RULE,
    RELU_RULE_1D,
    SIGMOID_RULE_1D,
    SIGMOID_RULE_2D,
    TANH_RULE_1D,
    TANH_RULE_2D,
)
from rextio_torch.claim.binops import (
    ADD_BROADCAST_2D_1D_RULE,
    ADD_SAME_RANK_RULE,
    MATMUL_BINOP_RULE,
    MATMUL_CALL_RULE,
    MUL_BROADCAST_2D_1D_RULE,
    MUL_SAME_RANK_RULE,
)
from rextio_torch.claim.linear import LINEAR_RULE
from rextio_torch.claim.reductions import MEAN_RULE, SUM_RULE
from rextio_torch.rules import torch_rule_records


def test_native_rules_match_claim_constants() -> None:
    by_id = {record.id: record for record in torch_rule_records()}
    for rule_id in (LINEAR_RULE, RELU_RULE, MEAN_RULE):
        record = by_id[rule_id]
        assert record.outcome == "native"
        assert record.stability == "experimental"
        # Phase A linear/relu-2d/mean are real-Cargo certified on the documented
        # CPython 3.11 / torch 2.11 macOS arm64 environment.
        assert record.verified is True
    for rule_id in (
        RELU_RULE_1D,
        SIGMOID_RULE_1D,
        SIGMOID_RULE_2D,
        TANH_RULE_1D,
        TANH_RULE_2D,
        MATMUL_BINOP_RULE,
        MATMUL_CALL_RULE,
        ADD_SAME_RANK_RULE,
        ADD_BROADCAST_2D_1D_RULE,
        MUL_SAME_RANK_RULE,
        MUL_BROADCAST_2D_1D_RULE,
        SUM_RULE,
    ):
        assert by_id[rule_id].outcome == "native"
        assert by_id[rule_id].verified is True


def test_diagnostic_codes_are_unique() -> None:
    codes = [record.diagnostic_code for record in torch_rule_records()]
    assert len(codes) == len(set(codes))


def test_unsupported_rejection_code_declared() -> None:
    codes = {record.diagnostic_code for record in torch_rule_records()}
    assert "RXTP-TORCH-010" in codes
    unsupported = next(
        record
        for record in torch_rule_records()
        if record.id == "rextio-torch/unsupported-tensor-surface"
    )
    assert unsupported.outcome == "fallback"
    assert unsupported.verified is False
