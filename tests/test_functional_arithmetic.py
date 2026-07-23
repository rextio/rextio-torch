"""Exact functional aliases for the bounded elementwise tensor matrix."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rextio.config.schema import RextioConfig
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

from rextio_torch.claim.binops import (
    FUNCTION_ADD_RULE,
    FUNCTION_DIV_RULE,
    FUNCTION_MUL_RULE,
    FUNCTION_SUB_RULE,
)
from rextio_torch.diagnostics import (
    DIAGNOSTIC_FUNCTION_ADD,
    DIAGNOSTIC_FUNCTION_DIV,
    DIAGNOSTIC_FUNCTION_MUL,
    DIAGNOSTIC_FUNCTION_SUB,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_torch.lower import binops as lower_binops
from rextio_torch.plugin import plugin
from rextio_torch.rust_snippets import (
    ADD,
    DIV,
    MUL,
    SUB,
    add_helper,
    div_helper,
    mul_helper,
    sub_helper,
)

PLUGIN = plugin()
CONFIG = RextioConfig()
NON_LITERAL = ClaimLiteral(is_literal=False)

FUNCTIONS = (
    ("torch.add", FUNCTION_ADD_RULE, DIAGNOSTIC_FUNCTION_ADD, ADD, add_helper()),
    ("torch.sub", FUNCTION_SUB_RULE, DIAGNOSTIC_FUNCTION_SUB, SUB, sub_helper()),
    ("torch.mul", FUNCTION_MUL_RULE, DIAGNOSTIC_FUNCTION_MUL, MUL, mul_helper()),
    ("torch.div", FUNCTION_DIV_RULE, DIAGNOSTIC_FUNCTION_DIV, DIV, div_helper()),
)
MATRICES = (
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D),
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
)


def _site(
    target: str,
    left: str | None = TENSOR_F32_CPU_2D,
    right: str | None = TENSOR_F32_CPU_2D,
    *,
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=(left, right),
        operand_literals=(NON_LITERAL, NON_LITERAL),
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
    )


def _context(
    *,
    operands: tuple[str, ...] = ("left", "right"),
    receiver: str | None = None,
) -> LoweringContext:
    return LoweringContext(
        operands=operands,
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
        receiver=receiver,
    )


@pytest.mark.parametrize(("target", "rule_id", "_diagnostic", "_call", "_helper"), FUNCTIONS)
@pytest.mark.parametrize(("left", "right", "result_type"), MATRICES)
def test_claims_exact_functional_tensor_matrix(
    target: str,
    rule_id: str,
    _diagnostic: str,
    _call: str,
    _helper: str,
    left: str,
    right: str,
    result_type: str,
) -> None:
    assert PLUGIN.claim(_site(target, left, right), CONFIG) == Claimed(
        rule_id=rule_id,
        result_type=result_type,
    )


@pytest.mark.parametrize(("target", "_rule", "diagnostic", "_call", "_helper"), FUNCTIONS)
def test_rejects_every_noncanonical_functional_arithmetic_shape(
    target: str,
    _rule: str,
    diagnostic: str,
    _call: str,
    _helper: str,
) -> None:
    option_name = {
        "torch.add": "alpha",
        "torch.sub": "alpha",
        "torch.mul": "out",
        "torch.div": "rounding_mode",
    }[target]
    option = KeywordArg(
        name=option_name,
        arg_type="str",
        literal=ClaimLiteral(is_literal=True, value="trunc"),
    )
    malformed = (
        _site(target, keywords=(option,)),
        replace(_site(target), operand_types=(TENSOR_F32_CPU_2D,)),
        replace(_site(target), operand_literals=(NON_LITERAL,)),
        replace(
            _site(target),
            operand_literals=(NON_LITERAL, ClaimLiteral(is_literal=True, value=1)),
        ),
        replace(
            _site(target),
            receiver=ReceiverMeta(
                arg_type=TENSOR_F32_CPU_2D,
                expr_kind="name",
                is_safe=True,
            ),
        ),
    )
    for candidate in malformed:
        result = PLUGIN.claim(candidate, CONFIG)
        assert isinstance(result, Rejected)
        assert result.diagnostic.code == diagnostic

    scalar = PLUGIN.claim(_site(target, TENSOR_F32_CPU_2D, "float"), CONFIG)
    assert isinstance(scalar, Rejected)
    assert scalar.diagnostic.code == DIAGNOSTIC_UNSUPPORTED

    classified = PLUGIN.claim(
        _site(target, TENSOR_F32_CPU_1D, TENSOR_I64_CPU_1D),
        CONFIG,
    )
    assert isinstance(classified, Rejected)
    assert classified.diagnostic.code == diagnostic


def test_noncanonical_function_names_remain_unclaimed() -> None:
    for target in ("torch.subtract", "torch.multiply", "torch.divide", "torch.Tensor.add"):
        assert isinstance(PLUGIN.claim(_site(target), CONFIG), NotCovered)


@pytest.mark.parametrize(
    ("target", "rule_id", "_diagnostic", "call_name", "helper"),
    FUNCTIONS,
)
@pytest.mark.parametrize(("left", "right", "result_type"), MATRICES)
def test_lowers_functional_arithmetic_through_existing_fallible_helpers(
    target: str,
    rule_id: str,
    _diagnostic: str,
    call_name: str,
    helper: str,
    left: str,
    right: str,
    result_type: str,
) -> None:
    claimed = replace(
        _site(target, left, right),
        rule_id=rule_id,
        result_type=result_type,
    )

    lowered = PLUGIN.lower(claimed, _context())

    assert lowered.rust == f"{call_name}(&left, &right)?"
    assert helper in lowered.helpers
    assert "no_grad_guard" in helper
    assert ".f_" in helper
    assert "unwrap(" not in helper


def test_lower_revalidates_functional_arithmetic_before_helper_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claimed = replace(
        _site("torch.add"),
        rule_id=FUNCTION_ADD_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )

    def unreachable() -> str:
        raise AssertionError("forged functional arithmetic reached helper generation")

    monkeypatch.setitem(
        lower_binops._FUNCTION_ELEMENTWISE,
        "torch.add",
        (FUNCTION_ADD_RULE, ADD, unreachable),
    )
    forged_sites = (
        replace(claimed, rule_id=FUNCTION_SUB_RULE),
        replace(
            claimed,
            receiver=ReceiverMeta(
                arg_type=TENSOR_F32_CPU_2D,
                expr_kind="name",
                is_safe=True,
            ),
        ),
        replace(claimed, keywords=(KeywordArg(name="alpha", arg_type="int"),)),
        replace(claimed, operand_types=(TENSOR_F32_CPU_2D,)),
        replace(claimed, operand_literals=(NON_LITERAL,)),
        replace(
            claimed,
            operand_literals=(NON_LITERAL, ClaimLiteral(is_literal=True, value=1)),
        ),
        replace(
            claimed,
            operand_types=(TENSOR_I64_CPU_1D, TENSOR_I64_CPU_1D),
            result_type=TENSOR_I64_CPU_1D,
        ),
        replace(claimed, result_type=TENSOR_F32_CPU_1D),
    )
    for forged in forged_sites:
        with pytest.raises(ValueError):
            PLUGIN.lower(forged, _context())

    for context in (
        _context(operands=("left",)),
        _context(operands=("left", "right", "extra")),
        _context(receiver="forged"),
    ):
        with pytest.raises(ValueError):
            PLUGIN.lower(claimed, context)
