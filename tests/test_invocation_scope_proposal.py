"""Core API 1.7 invocation-scope and guarded-helper contracts."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from rextio.plugins.api import (
    ClaimSite,
    LoweredExpr,
    LoweringContext,
    PLUGIN_API_VERSION,
    PluginFunctionScopeContext,
)

from rextio_torch.claim.linear import LINEAR_RULE, LINEAR_TARGET
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.invocation_scope import (
    CORE_17_LOWERED_EXPR_FIELDS,
    INVOCATION_SCOPE_OPTIMIZATION_ACTIVE,
    PROPOSAL_ID,
    PROPOSAL_STATUS,
    inspect_core_function_scope_capability,
    production_no_grad_baseline_summary,
    proposed_no_grad_function_body_support,
)
from rextio_torch.plugin import plugin
from rextio_torch.rust_snippets.ops import (
    FUNCTION_SCOPED_SUFFIX,
    add_helper,
    argmax_helper,
    div_helper,
    gelu_none_helper,
    linear_helper,
    linear_no_bias_helper,
    matmul_helper,
    mul_helper,
    reduction_helper,
    relu_helper,
    sigmoid_helper,
    softmax_helper,
    sub_helper,
    tanh_helper,
    unary_helper,
)

ROOT = Path(__file__).resolve().parents[1]


def _scope_context(
    *,
    used_rule_ids: tuple[str, ...] = (LINEAR_RULE,),
    has_python_boundary_calls: bool = False,
) -> PluginFunctionScopeContext:
    return PluginFunctionScopeContext(
        function_qualname="torch_app.kernels.inference",
        used_rule_ids=used_rule_ids,
        used_type_keys=(
            TENSOR_F32_CPU_1D,
            TENSOR_F32_CPU_2D,
        ),
        has_python_boundary_calls=has_python_boundary_calls,
    )


def _fresh_name(prefix: str) -> str:
    return f"{prefix}_0"


def _linear_claim() -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=LINEAR_TARGET,
        operand_types=(
            TENSOR_F32_CPU_2D,
            TENSOR_F32_CPU_2D,
            TENSOR_F32_CPU_1D,
        ),
        file_path="kernels.py",
        line=1,
        column=0,
        rule_id=LINEAR_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )


def test_core_api_17_function_scope_capability_is_active() -> None:
    assert PLUGIN_API_VERSION == "1.7"
    assert frozenset(LoweredExpr.__dataclass_fields__) == CORE_17_LOWERED_EXPR_FIELDS
    capability = inspect_core_function_scope_capability()
    assert capability.plugin_api_version == "1.7"
    assert "function_scope_guard" in capability.protocol_hook_names
    assert capability.has_function_body_scope_hook is True
    assert capability.invocation_scope_optimization_active is True
    assert INVOCATION_SCOPE_OPTIMIZATION_ACTIVE is True
    assert capability.proposal_id == PROPOSAL_ID
    assert capability.proposal_status == PROPOSAL_STATUS == "implemented-candidate"


def test_provider_guard_is_pyo3_claim_only_and_rxt075_fail_closed() -> None:
    provider = plugin()
    guard = provider.function_scope_guard(_scope_context())
    assert guard is not None
    assert guard.rust == "tch::no_grad_guard()"
    assert guard.uses == ()
    assert guard.helpers == ()

    assert (
        provider.function_scope_guard(
            _scope_context(has_python_boundary_calls=True)
        )
        is None
    )
    assert provider.function_scope_guard(_scope_context(used_rule_ids=())) is None

    standalone = cast(
        PluginFunctionScopeContext,
        SimpleNamespace(
            backend="standalone-rust",
            has_python_boundary_calls=False,
            used_rule_ids=(LINEAR_RULE,),
        ),
    )
    assert provider.function_scope_guard(standalone) is None

    with pytest.raises(
        ValueError,
        match="has_python_boundary_calls must be a bool",
    ):
        PluginFunctionScopeContext(
            function_qualname="torch_app.kernels.invalid",
            used_rule_ids=(LINEAR_RULE,),
            used_type_keys=(TENSOR_F32_CPU_2D,),
            has_python_boundary_calls=1,  # type: ignore[arg-type]
        )


def test_every_operation_has_distinct_guarded_and_function_scoped_helpers() -> None:
    variants = (
        (linear_helper(), linear_helper(function_scope_guard_active=True)),
        (
            linear_no_bias_helper(),
            linear_no_bias_helper(function_scope_guard_active=True),
        ),
        (relu_helper(), relu_helper(function_scope_guard_active=True)),
        (sigmoid_helper(), sigmoid_helper(function_scope_guard_active=True)),
        (tanh_helper(), tanh_helper(function_scope_guard_active=True)),
        (unary_helper("abs"), unary_helper("abs", function_scope_guard_active=True)),
        (gelu_none_helper(), gelu_none_helper(function_scope_guard_active=True)),
        (
            reduction_helper("mean", 1, False),
            reduction_helper(
                "mean",
                1,
                False,
                function_scope_guard_active=True,
            ),
        ),
        (add_helper(), add_helper(function_scope_guard_active=True)),
        (mul_helper(), mul_helper(function_scope_guard_active=True)),
        (sub_helper(), sub_helper(function_scope_guard_active=True)),
        (div_helper(), div_helper(function_scope_guard_active=True)),
        (matmul_helper(), matmul_helper(function_scope_guard_active=True)),
        (softmax_helper(1), softmax_helper(1, function_scope_guard_active=True)),
        (
            argmax_helper(1, False, expected_rank=1),
            argmax_helper(
                1,
                False,
                expected_rank=1,
                function_scope_guard_active=True,
            ),
        ),
    )
    for guarded, scoped in variants:
        assert guarded.count("tch::no_grad_guard()") == 1
        assert "tch::no_grad_guard()" not in scoped
        guarded_name = re.search(r"\bfn\s+([A-Za-z0-9_]+)", guarded)
        scoped_name = re.search(r"\bfn\s+([A-Za-z0-9_]+)", scoped)
        assert guarded_name is not None
        assert scoped_name is not None
        assert guarded_name.group(1) != scoped_name.group(1)
        assert scoped_name.group(1) == guarded_name.group(1) + FUNCTION_SCOPED_SUFFIX


def test_lowering_uses_scoped_variant_only_for_explicit_core_signal() -> None:
    provider = plugin()
    active = LoweringContext(
        operands=("x", "w", "b"),
        target_language="rust",
        fresh_name=_fresh_name,
        function_scope_guard_active=True,
    )
    active_lowered = provider.lower(_linear_claim(), active)
    assert active_lowered.rust.startswith("__rxttorch_linear_function_scoped(")
    assert any(
        "fn __rxttorch_linear_function_scoped(" in helper
        and "no_grad_guard" not in helper
        for helper in active_lowered.helpers
    )

    legacy = cast(
        LoweringContext,
        SimpleNamespace(
            operands=("x", "w", "b"),
            fresh_name=_fresh_name,
            receiver=None,
        ),
    )
    legacy_lowered = provider.lower(_linear_claim(), legacy)
    assert legacy_lowered.rust.startswith("__rxttorch_linear(")
    assert any(
        "fn __rxttorch_linear(" in helper and "no_grad_guard" in helper
        for helper in legacy_lowered.helpers
    )

    for unexpected_truthy in (1, "yes"):
        malformed = cast(
            LoweringContext,
            SimpleNamespace(
                operands=("x", "w", "b"),
                fresh_name=_fresh_name,
                receiver=None,
                function_scope_guard_active=unexpected_truthy,
            ),
        )
        malformed_lowered = provider.lower(_linear_claim(), malformed)
        assert malformed_lowered.rust.startswith("__rxttorch_linear(")
        assert "_function_scoped" not in malformed_lowered.rust


def test_historical_proposal_payload_is_stack_raii_only() -> None:
    support = proposed_no_grad_function_body_support()
    assert support.rust_prelude == (
        "let _rxttorch_invocation_no_grad = tch::no_grad_guard();",
    )
    assert support.rust_epilogue == ()
    joined = "\n".join(support.rust_prelude)
    assert "thread_local" not in joined
    assert "static " not in joined


def test_status_document_describes_active_and_fail_closed_paths() -> None:
    path = ROOT / "docs" / "invocation-scope-proposal-0.1.3.md"
    text = path.read_text(encoding="utf-8")
    assert "implemented-candidate" in text
    assert "function_scope_guard" in text
    assert "RXT075" in text
    assert "per-operation" in text.lower()
    summary = production_no_grad_baseline_summary()
    assert "eligible native PyO3 functions" in summary
    assert "active=True" in summary
