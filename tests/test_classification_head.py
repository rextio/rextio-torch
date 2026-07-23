"""Claim/lower/fallback checks for the narrow Torch classification head."""

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

from rextio_torch.claim.classification import ARGMAX_RULE, SOFTMAX_RULE
from rextio_torch.diagnostics import (
    DIAGNOSTIC_ARGMAX,
    DIAGNOSTIC_SOFTMAX,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_torch.plugin import plugin
from rextio_torch.rust_snippets import (
    ARGMAX_DIM1_KEEPFALSE,
    SOFTMAX_DIM1,
    argmax_dim1_keepfalse_helper,
    softmax_dim1_helper,
)

PLUGIN = plugin()
CONFIG = RextioConfig()


def _literal(name: str, value: object) -> KeywordArg:
    return KeywordArg(
        name=name,
        arg_type=type(value).__name__,
        literal=ClaimLiteral(is_literal=True, value=value),
    )


def _site(method: str, keywords: tuple[KeywordArg, ...], *, receiver: str | None = TENSOR_F32_CPU_2D):
    return ClaimSite(
        kind="call",
        target=f"tensor.{method}",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        keywords=keywords,
        receiver=(
            ReceiverMeta(arg_type=receiver, expr_kind="call", is_safe=False)
            if receiver is not None
            else None
        ),
    )


def _fresh_name(prefix: str) -> str:
    return f"{prefix}_0"


def test_claims_method_only_classification_head() -> None:
    softmax = PLUGIN.claim(_site("softmax", (_literal("dim", 1),)), CONFIG)
    assert softmax == Claimed(rule_id=SOFTMAX_RULE, result_type=TENSOR_F32_CPU_2D)
    argmax = PLUGIN.claim(
        _site("argmax", (_literal("dim", 1), _literal("keepdim", False))), CONFIG
    )
    assert argmax == Claimed(rule_id=ARGMAX_RULE, result_type=TENSOR_I64_CPU_1D)


@pytest.mark.parametrize(
    ("method", "keywords", "diagnostic"),
    [
        ("softmax", (_literal("dim", 0),), DIAGNOSTIC_SOFTMAX),
        ("softmax", (_literal("dim", 1), _literal("dtype", "float32")), DIAGNOSTIC_SOFTMAX),
        ("argmax", (_literal("dim", 1), _literal("keepdim", True)), DIAGNOSTIC_ARGMAX),
        ("argmax", (_literal("keepdim", False),), DIAGNOSTIC_ARGMAX),
    ],
)
def test_rejects_classification_near_misses(
    method: str, keywords: tuple[KeywordArg, ...], diagnostic: str
) -> None:
    result = PLUGIN.claim(_site(method, keywords), CONFIG)
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == diagnostic


def test_dynamic_or_functional_forms_fall_back() -> None:
    dynamic = KeywordArg(
        name="dim", arg_type="int", literal=ClaimLiteral(is_literal=False, value=None)
    )
    assert isinstance(PLUGIN.claim(_site("softmax", (dynamic,)), CONFIG), Rejected)
    functional = ClaimSite(
        kind="call",
        target="torch.softmax",
        operand_types=(TENSOR_F32_CPU_2D,),
        file_path="",
        line=0,
        column=0,
        keywords=(_literal("dim", 1),),
    )
    assert isinstance(PLUGIN.claim(functional, CONFIG), NotCovered)


def test_lower_classification_helpers_revalidate_exact_metadata() -> None:
    for method, rule, result_type, keywords, call_name, helper in (
        ("softmax", SOFTMAX_RULE, TENSOR_F32_CPU_2D, (_literal("dim", 1),), SOFTMAX_DIM1, softmax_dim1_helper()),
        (
            "argmax",
            ARGMAX_RULE,
            TENSOR_I64_CPU_1D,
            (_literal("dim", 1), _literal("keepdim", False)),
            ARGMAX_DIM1_KEEPFALSE,
            argmax_dim1_keepfalse_helper(),
        ),
    ):
        claimed = _site(method, keywords)
        claimed = replace(claimed, rule_id=rule, result_type=result_type)
        lowered = PLUGIN.lower(
            claimed,
            LoweringContext(
                operands=(), target_language="rust", fresh_name=_fresh_name, receiver="logits"
            ),
        )
        assert lowered.rust == f"{call_name}(&logits)?"
        assert helper in lowered.helpers
        assert "no_grad_guard" in helper

    malformed = _site("argmax", (_literal("dim", 1), _literal("keepdim", True)))
    malformed = replace(malformed, rule_id=ARGMAX_RULE, result_type=TENSOR_I64_CPU_1D)
    with pytest.raises(ValueError, match="keepdim=False"):
        PLUGIN.lower(
            malformed,
            LoweringContext(
                operands=(), target_language="rust", fresh_name=_fresh_name, receiver="logits"
            ),
        )
