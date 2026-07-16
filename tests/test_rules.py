"""Rule surface integrity for Phase A."""

from __future__ import annotations

from rextio_torch.claim.activations import RELU_RULE
from rextio_torch.claim.linear import LINEAR_RULE
from rextio_torch.claim.reductions import MEAN_RULE
from rextio_torch.rules import torch_rule_records


def test_native_rules_match_claim_constants() -> None:
    by_id = {record.id: record for record in torch_rule_records()}
    for rule_id in (LINEAR_RULE, RELU_RULE, MEAN_RULE):
        record = by_id[rule_id]
        assert record.outcome == "native"
        assert record.stability == "experimental"
        # Phase A is real-Cargo certified on the documented CPython 3.11 /
        # torch 2.11 macOS arm64 environment.
        assert record.verified is True


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
