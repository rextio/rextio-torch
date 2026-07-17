"""Focused claim tests for the Alpha AOT tensor surface."""

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

from rextio_torch.claim.activations import (
    RELU_RULE,
    RELU_RULE_1D,
    SIGMOID_RULE_2D,
    TANH_RULE_2D,
)
from rextio_torch.claim.binops import (
    ADD_BROADCAST_2D_1D_RULE,
    ADD_SAME_RANK_RULE,
    MATMUL_BINOP_RULE,
    MATMUL_CALL_RULE,
    MATMUL_CALL_TARGET,
)
from rextio_torch.claim.linear import LINEAR_RULE, LINEAR_TARGET
from rextio_torch.claim.reductions import MEAN_RULE, SUM_RULE
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


def _binop_site(op: str, left: str | None, right: str | None) -> ClaimSite:
    return ClaimSite(
        kind="binop",
        target=op,
        operand_types=(left, right),
        file_path="",
        line=0,
        column=0,
    )


def test_claims_functional_linear_phase_a() -> None:
    result = PLUGIN.claim(_linear_site(), CONFIG)
    assert result == Claimed(rule_id=LINEAR_RULE, result_type=TENSOR_F32_CPU_2D)


def test_claims_relu_on_rank2() -> None:
    result = PLUGIN.claim(_method_site("relu", TENSOR_F32_CPU_2D), CONFIG)
    assert result == Claimed(rule_id=RELU_RULE, result_type=TENSOR_F32_CPU_2D)


def test_claims_relu_on_rank1() -> None:
    result = PLUGIN.claim(_method_site("relu", TENSOR_F32_CPU_1D), CONFIG)
    assert result == Claimed(rule_id=RELU_RULE_1D, result_type=TENSOR_F32_CPU_1D)


def test_claims_sigmoid_and_tanh_rank2() -> None:
    assert PLUGIN.claim(_method_site("sigmoid", TENSOR_F32_CPU_2D), CONFIG) == Claimed(
        rule_id=SIGMOID_RULE_2D, result_type=TENSOR_F32_CPU_2D
    )
    assert PLUGIN.claim(_method_site("tanh", TENSOR_F32_CPU_2D), CONFIG) == Claimed(
        rule_id=TANH_RULE_2D, result_type=TENSOR_F32_CPU_2D
    )


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


def test_claims_sum_dim1_keepdim_false() -> None:
    keywords = (
        KeywordArg(name="dim", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
        KeywordArg(
            name="keepdim",
            arg_type="bool",
            literal=ClaimLiteral(is_literal=True, value=False),
        ),
    )
    result = PLUGIN.claim(
        _method_site("sum", TENSOR_F32_CPU_2D, keywords=keywords),
        CONFIG,
    )
    assert result == Claimed(rule_id=SUM_RULE, result_type=TENSOR_F32_CPU_1D)


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


def test_claims_same_rank_add() -> None:
    assert PLUGIN.claim(_binop_site("+", TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D), CONFIG) == Claimed(
        rule_id=ADD_SAME_RANK_RULE, result_type=TENSOR_F32_CPU_2D
    )
    assert PLUGIN.claim(_binop_site("+", TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D), CONFIG) == Claimed(
        rule_id=ADD_SAME_RANK_RULE, result_type=TENSOR_F32_CPU_1D
    )


def test_claims_broadcast_add_rank2_rank1() -> None:
    assert PLUGIN.claim(_binop_site("+", TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D), CONFIG) == Claimed(
        rule_id=ADD_BROADCAST_2D_1D_RULE, result_type=TENSOR_F32_CPU_2D
    )
    assert PLUGIN.claim(_binop_site("+", TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D), CONFIG) == Claimed(
        rule_id=ADD_BROADCAST_2D_1D_RULE, result_type=TENSOR_F32_CPU_2D
    )


def test_claims_matmul_binop_and_call() -> None:
    assert PLUGIN.claim(_binop_site("@", TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D), CONFIG) == Claimed(
        rule_id=MATMUL_BINOP_RULE, result_type=TENSOR_F32_CPU_2D
    )
    site = ClaimSite(
        kind="call",
        target=MATMUL_CALL_TARGET,
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
    )
    assert PLUGIN.claim(site, CONFIG) == Claimed(
        rule_id=MATMUL_CALL_RULE, result_type=TENSOR_F32_CPU_2D
    )

    method = _method_site(
        "matmul",
        TENSOR_F32_CPU_2D,
        operand_types=(TENSOR_F32_CPU_2D,),
    )
    assert PLUGIN.claim(method, CONFIG) == Claimed(
        rule_id=MATMUL_CALL_RULE, result_type=TENSOR_F32_CPU_2D
    )


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


def test_rejects_matmul_rank1() -> None:
    result = PLUGIN.claim(_binop_site("@", TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D), CONFIG)
    assert isinstance(result, Rejected)


def test_not_covered_for_unrelated_target() -> None:
    site = ClaimSite(
        kind="call",
        target="torch.softmax",
        operand_types=(TENSOR_F32_CPU_2D,),
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
