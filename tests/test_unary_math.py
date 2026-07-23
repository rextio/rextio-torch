"""Bounded exact unary math spellings for float32 CPU rank-1/rank-2 tensors."""

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

from rextio_torch.diagnostics import (
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_torch.plugin import plugin

PLUGIN = plugin()
CONFIG = RextioConfig()
NON_LITERAL = ClaimLiteral(is_literal=False)

OPERATIONS = (
    (
        "abs",
        ("torch.abs",),
        ("abs",),
        "rextio-torch/unary-abs-f32-cpu-rank1-2",
        "__rxttorch_abs",
        "f_abs",
    ),
    (
        "neg",
        ("torch.neg",),
        ("neg",),
        "rextio-torch/unary-neg-f32-cpu-rank1-2",
        "__rxttorch_neg",
        "f_neg",
    ),
    (
        "negative",
        ("torch.negative",),
        ("negative",),
        "rextio-torch/unary-negative-f32-cpu-rank1-2",
        "__rxttorch_negative",
        "f_negative",
    ),
    (
        "square",
        ("torch.square",),
        ("square",),
        "rextio-torch/unary-square-f32-cpu-rank1-2",
        "__rxttorch_square",
        "f_square",
    ),
    (
        "exp",
        ("torch.exp",),
        ("exp",),
        "rextio-torch/unary-exp-f32-cpu-rank1-2",
        "__rxttorch_exp",
        "f_exp",
    ),
    (
        "log",
        ("torch.log",),
        ("log",),
        "rextio-torch/unary-log-f32-cpu-rank1-2",
        "__rxttorch_log",
        "f_log",
    ),
    (
        "sqrt",
        ("torch.sqrt",),
        ("sqrt",),
        "rextio-torch/unary-sqrt-f32-cpu-rank1-2",
        "__rxttorch_sqrt",
        "f_sqrt",
    ),
)


def _function(target: str, arg_type: str | None = TENSOR_F32_CPU_2D) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=(arg_type,),
        operand_literals=(NON_LITERAL,),
        file_path="",
        line=0,
        column=0,
    )


def _method(method: str, receiver_type: str | None = TENSOR_F32_CPU_2D) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=f"torch.Tensor.{method}",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        receiver=ReceiverMeta(
            arg_type=receiver_type,
            expr_kind="name",
            is_safe=True,
        ),
    )


def _context(
    *,
    operands: tuple[str, ...] = (),
    receiver: str | None = None,
) -> LoweringContext:
    return LoweringContext(
        operands=operands,
        receiver=receiver,
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )


@pytest.mark.parametrize(("_op", "functions", "methods", "rule_id", "_symbol", "_fallible"), OPERATIONS)
@pytest.mark.parametrize("rank_type", (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D))
def test_claims_every_exact_unary_function_and_method_alias(
    _op: str,
    functions: tuple[str, ...],
    methods: tuple[str, ...],
    rule_id: str,
    _symbol: str,
    _fallible: str,
    rank_type: str,
) -> None:
    expected = Claimed(rule_id=rule_id, result_type=rank_type)
    for target in functions:
        assert PLUGIN.claim(_function(target, rank_type), CONFIG) == expected
    for method in methods:
        assert PLUGIN.claim(_method(method, rank_type), CONFIG) == expected


@pytest.mark.parametrize(("_op", "functions", "methods", "_rule", "_symbol", "_fallible"), OPERATIONS)
def test_rejects_noncanonical_unary_call_shapes(
    _op: str,
    functions: tuple[str, ...],
    methods: tuple[str, ...],
    _rule: str,
    _symbol: str,
    _fallible: str,
) -> None:
    keyword = KeywordArg(
        name="out",
        arg_type=TENSOR_F32_CPU_2D,
        literal=NON_LITERAL,
    )
    for target in functions:
        valid = _function(target)
        malformed = (
            replace(valid, keywords=(keyword,)),
            replace(valid, operand_types=(), operand_literals=()),
            replace(valid, operand_literals=()),
            replace(valid, operand_literals=(NON_LITERAL, NON_LITERAL)),
            replace(
                valid,
                operand_literals=(ClaimLiteral(is_literal=True, value=1),),
            ),
            replace(
                valid,
                operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
                operand_literals=(NON_LITERAL, NON_LITERAL),
            ),
        )
        for site in malformed:
            assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)

        scalar = PLUGIN.claim(_function(target, "float"), CONFIG)
        assert isinstance(scalar, Rejected)
        assert scalar.diagnostic.code == DIAGNOSTIC_UNSUPPORTED

        classified = PLUGIN.claim(_function(target, TENSOR_I64_CPU_1D), CONFIG)
        assert isinstance(classified, Rejected)
        assert classified.diagnostic.code == DIAGNOSTIC_UNSUPPORTED

    for method in methods:
        valid = _method(method)
        malformed = (
            replace(valid, operand_types=(TENSOR_F32_CPU_2D,)),
            replace(valid, operand_literals=(NON_LITERAL,)),
            replace(valid, keywords=(keyword,)),
        )
        for site in malformed:
            assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)


def test_unlisted_unary_aliases_remain_unclaimed() -> None:
    for target in ("torch.absolute", "torch.positive", "torch.absolute_", "torch.linalg.sqrt"):
        assert isinstance(PLUGIN.claim(_function(target), CONFIG), NotCovered)
    for method in ("absolute", "positive", "abs_", "neg_", "sqrt_"):
        assert isinstance(PLUGIN.claim(_method(method), CONFIG), NotCovered)


def test_unary_methods_preserve_existing_receiver_metadata_policy() -> None:
    receiver = ReceiverMeta(
        arg_type=TENSOR_F32_CPU_2D,
        expr_kind="call",
        is_safe=False,
    )
    site = replace(_method("sqrt"), receiver=receiver)
    assert PLUGIN.claim(site, CONFIG) == Claimed(
        rule_id="rextio-torch/unary-sqrt-f32-cpu-rank1-2",
        result_type=TENSOR_F32_CPU_2D,
    )
    claimed = replace(
        site,
        rule_id="rextio-torch/unary-sqrt-f32-cpu-rank1-2",
        result_type=TENSOR_F32_CPU_2D,
    )
    assert PLUGIN.lower(claimed, _context(receiver="input")).rust == (
        "__rxttorch_sqrt(&input)?"
    )


@pytest.mark.parametrize(("_op", "functions", "methods", "rule_id", "symbol", "fallible"), OPERATIONS)
def test_lowers_unary_aliases_through_fallible_no_grad_helpers(
    _op: str,
    functions: tuple[str, ...],
    methods: tuple[str, ...],
    rule_id: str,
    symbol: str,
    fallible: str,
) -> None:
    for target in functions:
        claimed = replace(
            _function(target),
            rule_id=rule_id,
            result_type=TENSOR_F32_CPU_2D,
        )
        lowered = PLUGIN.lower(claimed, _context(operands=("input",)))
        assert lowered.rust == f"{symbol}(&input)?"
        helper = next(item for item in lowered.helpers if f"fn {symbol}" in item)
        assert fallible in helper
        assert "no_grad_guard" in helper
        assert "unwrap(" not in helper
        assert "panic!" not in helper

    for method in methods:
        claimed = replace(
            _method(method),
            rule_id=rule_id,
            result_type=TENSOR_F32_CPU_2D,
        )
        lowered = PLUGIN.lower(claimed, _context(receiver="input"))
        assert lowered.rust == f"{symbol}(&input)?"


def test_lower_revalidates_unary_rule_alias_shape_and_result_metadata() -> None:
    valid_function = replace(
        _function("torch.sqrt"),
        rule_id="rextio-torch/unary-sqrt-f32-cpu-rank1-2",
        result_type=TENSOR_F32_CPU_2D,
    )
    valid_method = replace(
        _method("negative"),
        rule_id="rextio-torch/unary-negative-f32-cpu-rank1-2",
        result_type=TENSOR_F32_CPU_2D,
    )
    forged_functions = (
        replace(valid_function, rule_id="rextio-torch/unary-log-f32-cpu-rank1-2"),
        replace(valid_function, target="torch.absolute"),
        replace(valid_function, result_type=TENSOR_F32_CPU_1D),
        replace(valid_function, operand_types=(TENSOR_I64_CPU_1D,)),
        replace(valid_function, operand_literals=()),
        replace(valid_function, operand_literals=(NON_LITERAL, NON_LITERAL)),
        replace(
            valid_function,
            operand_literals=(ClaimLiteral(is_literal=True, value=1),),
        ),
        replace(
            valid_function,
            receiver=ReceiverMeta(
                arg_type=TENSOR_F32_CPU_2D,
                expr_kind="name",
                is_safe=True,
            ),
        ),
    )
    for forged in forged_functions:
        with pytest.raises(ValueError):
            PLUGIN.lower(forged, _context(operands=("input",)))

    forged_methods = (
        replace(valid_method, rule_id="rextio-torch/unary-abs-f32-cpu-rank1-2"),
        replace(valid_method, operand_types=(TENSOR_F32_CPU_2D,)),
        replace(valid_method, operand_literals=(NON_LITERAL,)),
        replace(valid_method, receiver=None),
        replace(valid_method, result_type=TENSOR_F32_CPU_1D),
    )
    for forged in forged_methods:
        with pytest.raises(ValueError):
            PLUGIN.lower(forged, _context(receiver="input"))

    for site, context in (
        (valid_function, _context()),
        (valid_function, _context(operands=("input",), receiver="forged")),
        (valid_method, _context()),
        (valid_method, _context(receiver="input", operands=("forged",))),
    ):
        with pytest.raises(ValueError):
            PLUGIN.lower(site, context)
