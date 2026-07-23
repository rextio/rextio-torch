"""Exact bounded functional GELU with fixed ``approximate='none'``."""

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
TARGET = "torch.nn.functional.gelu"
RULE = "rextio-torch/functional-gelu-none-f32-cpu-rank1-2"
NON_LITERAL = ClaimLiteral(is_literal=False)


def _keyword(
    value: object = "none",
    *,
    is_literal: bool = True,
    name: str = "approximate",
    arg_type: str = "str",
) -> KeywordArg:
    return KeywordArg(
        name=name,
        arg_type=arg_type,
        literal=ClaimLiteral(
            is_literal=is_literal,
            value=value if is_literal else None,
        ),
    )


def _site(
    arg_type: str | None = TENSOR_F32_CPU_2D,
    *,
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=TARGET,
        operand_types=(arg_type,),
        operand_literals=(NON_LITERAL,),
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
    )


def _context(
    *,
    operands: tuple[str, ...] = ("input",),
    receiver: str | None = None,
) -> LoweringContext:
    return LoweringContext(
        operands=operands,
        receiver=receiver,
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
    )


@pytest.mark.parametrize("rank_type", (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D))
def test_claims_default_and_explicit_literal_none_gelu(rank_type: str) -> None:
    expected = Claimed(rule_id=RULE, result_type=rank_type)
    assert PLUGIN.claim(_site(rank_type), CONFIG) == expected
    assert PLUGIN.claim(
        _site(rank_type, keywords=(_keyword("none"),)),
        CONFIG,
    ) == expected


def test_rejects_nonexact_gelu_options_and_call_shapes() -> None:
    malformed = (
        _site(keywords=(_keyword("tanh"),)),
        _site(keywords=(_keyword("none", is_literal=False),)),
        _site(keywords=(_keyword("none", name="out"),)),
        _site(keywords=(_keyword("none", arg_type="object"),)),
        replace(
            _site(),
            operand_types=(TENSOR_F32_CPU_2D, "str"),
            operand_literals=(
                NON_LITERAL,
                ClaimLiteral(is_literal=True, value="none"),
            ),
        ),
        replace(_site(), operand_types=(), operand_literals=()),
        replace(
            _site(),
            receiver=ReceiverMeta(
                arg_type=TENSOR_F32_CPU_2D,
                expr_kind="name",
                is_safe=True,
            ),
        ),
    )
    for site in malformed:
        rejected = PLUGIN.claim(site, CONFIG)
        assert isinstance(rejected, Rejected)
        assert rejected.diagnostic.code == "RXTP-TORCH-044"

    for arg_type in ("float", TENSOR_I64_CPU_1D):
        rejected = PLUGIN.claim(_site(arg_type), CONFIG)
        assert isinstance(rejected, Rejected)
        assert rejected.diagnostic.code == DIAGNOSTIC_UNSUPPORTED

    assert isinstance(PLUGIN.claim(_site(None), CONFIG), NotCovered)


def test_module_method_and_approximate_aliases_remain_unclaimed() -> None:
    method = replace(
        _site(),
        target="torch.Tensor.gelu",
        operand_types=(),
        operand_literals=(),
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CPU_2D,
            expr_kind="name",
            is_safe=True,
        ),
    )
    module = replace(_site(), target="torch.nn.GELU")
    for site in (method, module):
        assert isinstance(PLUGIN.claim(site, CONFIG), NotCovered)


@pytest.mark.parametrize("keywords", ((), (_keyword("none"),)))
def test_lowers_gelu_to_fixed_fallible_none_helper(
    keywords: tuple[KeywordArg, ...],
) -> None:
    claimed = replace(
        _site(keywords=keywords),
        rule_id=RULE,
        result_type=TENSOR_F32_CPU_2D,
    )

    lowered = PLUGIN.lower(claimed, _context())

    assert lowered.rust == "__rxttorch_gelu_none(&input)?"
    helper = next(item for item in lowered.helpers if "fn __rxttorch_gelu_none" in item)
    assert '.f_gelu("none")' in helper
    assert '"tanh"' not in helper
    assert "no_grad_guard" in helper
    assert "map_err(__rxttorch_map_err)" in helper
    assert "unwrap(" not in helper
    assert "panic!" not in helper


def test_lower_revalidates_gelu_target_option_type_value_and_context() -> None:
    valid = replace(
        _site(keywords=(_keyword("none"),)),
        rule_id=RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    forged = (
        replace(valid, rule_id="rextio-torch/unary-exp-f32-cpu-rank1-2"),
        replace(valid, target="torch.gelu"),
        replace(valid, keywords=(_keyword("tanh"),)),
        replace(valid, keywords=(_keyword("none", is_literal=False),)),
        replace(valid, keywords=(_keyword("none", arg_type="object"),)),
        replace(valid, result_type=TENSOR_F32_CPU_1D),
        replace(valid, operand_types=(TENSOR_I64_CPU_1D,)),
        replace(
            valid,
            receiver=ReceiverMeta(
                arg_type=TENSOR_F32_CPU_2D,
                expr_kind="name",
                is_safe=True,
            ),
        ),
    )
    for site in forged:
        with pytest.raises(ValueError):
            PLUGIN.lower(site, _context())

    for context in (
        _context(operands=()),
        _context(operands=("input", "extra")),
        _context(receiver="forged"),
    ):
        with pytest.raises(ValueError):
            PLUGIN.lower(valid, context)
