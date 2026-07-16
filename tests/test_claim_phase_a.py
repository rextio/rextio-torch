"""Focused claim tests for the linear → ReLU → mean Phase A slice."""

from __future__ import annotations

from rextio.config.schema import RextioConfig
from rextio.plugins.api import (
    ClaimLiteral,
    Claimed,
    ClaimSite,
    KeywordArg,
    NotCovered,
    ReceiverMeta,
    Rejected,
)

from rextio_torch.claim.activations import RELU_RULE
from rextio_torch.claim.linear import LINEAR_RULE, LINEAR_TARGET
from rextio_torch.claim.reductions import MEAN_RULE
from rextio_torch.diagnostics import (
    DIAGNOSTIC_MEAN,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
)
from rextio_torch.plugin import plugin

PLUGIN = plugin()
CONFIG = RextioConfig()


def _linear_site(
    operands: tuple[str | None, ...] = (
        TENSOR_F32_CPU_2D,
        TENSOR_F32_CPU_2D,
        TENSOR_F32_CPU_1D,
    ),
    *,
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=LINEAR_TARGET,
        operand_types=operands,
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
    )


def _method_site(
    method: str,
    receiver_type: str | None,
    *,
    keywords: tuple[KeywordArg, ...] = (),
    operand_types: tuple[str | None, ...] = (),
    target: str | None = None,
) -> ClaimSite:
    receiver = None
    if receiver_type is not None or target is None:
        # Method form: receiver present when typed; allow unresolved None type.
        receiver = (
            ReceiverMeta(arg_type=receiver_type, expr_kind="name", is_safe=True)
            if receiver_type is not None
            else ReceiverMeta(arg_type=None, expr_kind="name", is_safe=True)
        )
    return ClaimSite(
        kind="call",
        target=target if target is not None else f"tensor.{method}",
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
        receiver=receiver,
    )


def test_claims_functional_linear_phase_a() -> None:
    result = PLUGIN.claim(_linear_site(), CONFIG)
    assert result == Claimed(rule_id=LINEAR_RULE, result_type=TENSOR_F32_CPU_2D)


def test_claims_relu_on_rank2() -> None:
    result = PLUGIN.claim(_method_site("relu", TENSOR_F32_CPU_2D), CONFIG)
    assert result == Claimed(rule_id=RELU_RULE, result_type=TENSOR_F32_CPU_2D)


def test_claims_relu_with_bare_method_target() -> None:
    # Intermediate chain receivers may present target as the bare method name.
    result = PLUGIN.claim(
        _method_site("relu", TENSOR_F32_CPU_2D, target="relu"),
        CONFIG,
    )
    assert result == Claimed(rule_id=RELU_RULE, result_type=TENSOR_F32_CPU_2D)


def test_claims_mean_dim1_keepdim_false() -> None:
    keywords = (
        KeywordArg(name="dim", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
        KeywordArg(
            name="keepdim",
            arg_type="bool",
            literal=ClaimLiteral(is_literal=True, value=False),
        ),
    )
    result = PLUGIN.claim(
        _method_site("mean", TENSOR_F32_CPU_2D, keywords=keywords),
        CONFIG,
    )
    assert result == Claimed(rule_id=MEAN_RULE, result_type=TENSOR_F32_CPU_1D)


def test_mean_keyword_order_is_irrelevant() -> None:
    keywords = (
        KeywordArg(
            name="keepdim",
            arg_type="bool",
            literal=ClaimLiteral(is_literal=True, value=False),
        ),
        KeywordArg(name="dim", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
    )
    result = PLUGIN.claim(
        _method_site("mean", TENSOR_F32_CPU_2D, keywords=keywords),
        CONFIG,
    )
    assert result == Claimed(rule_id=MEAN_RULE, result_type=TENSOR_F32_CPU_1D)


def test_rejects_wrong_linear_ranks() -> None:
    result = PLUGIN.claim(
        _linear_site((TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D)),
        CONFIG,
    )
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == DIAGNOSTIC_UNSUPPORTED


def test_rejects_mean_wrong_dim() -> None:
    keywords = (
        KeywordArg(name="dim", arg_type="int", literal=ClaimLiteral(is_literal=True, value=0)),
        KeywordArg(
            name="keepdim",
            arg_type="bool",
            literal=ClaimLiteral(is_literal=True, value=False),
        ),
    )
    result = PLUGIN.claim(
        _method_site("mean", TENSOR_F32_CPU_2D, keywords=keywords),
        CONFIG,
    )
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == DIAGNOSTIC_MEAN


def test_rejects_mean_keepdim_true() -> None:
    keywords = (
        KeywordArg(name="dim", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
        KeywordArg(
            name="keepdim",
            arg_type="bool",
            literal=ClaimLiteral(is_literal=True, value=True),
        ),
    )
    result = PLUGIN.claim(
        _method_site("mean", TENSOR_F32_CPU_2D, keywords=keywords),
        CONFIG,
    )
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == DIAGNOSTIC_MEAN


def test_rejects_relu_on_rank1() -> None:
    result = PLUGIN.claim(_method_site("relu", TENSOR_F32_CPU_1D), CONFIG)
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == DIAGNOSTIC_UNSUPPORTED


def test_not_covered_for_unrelated_target() -> None:
    site = ClaimSite(
        kind="call",
        target="torch.matmul",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
    )
    assert isinstance(PLUGIN.claim(site, CONFIG), NotCovered)


def test_not_covered_for_unresolved_linear_operands() -> None:
    result = PLUGIN.claim(
        _linear_site((TENSOR_F32_CPU_2D, None, TENSOR_F32_CPU_1D)),
        CONFIG,
    )
    assert isinstance(result, NotCovered)
