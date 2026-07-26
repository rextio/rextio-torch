"""Core capability absence and invocation-scope proposal contracts."""

from __future__ import annotations

from pathlib import Path

from rextio.plugins.api import LoweredExpr, PLUGIN_API_VERSION

from rextio_torch.invocation_scope import (
    CORE_16_LOWERED_EXPR_FIELDS,
    INVOCATION_SCOPE_OPTIMIZATION_ACTIVE,
    PROPOSED_FUNCTION_SCOPE_HOOK_NAMES,
    PROPOSAL_ID,
    PROPOSAL_STATUS,
    inspect_core_function_scope_capability,
    production_no_grad_baseline_summary,
    proposed_no_grad_function_body_support,
)
from rextio_torch.plugin import plugin
from rextio_torch.rust_snippets.ops import (
    argmax_dim1_keepfalse_helper,
    linear_helper,
    relu_helper,
    softmax_dim1_helper,
    sub_helper,
    tanh_helper,
)

ROOT = Path(__file__).resolve().parents[1]


def test_core_lowered_expr_is_expression_only() -> None:
    fields = frozenset(LoweredExpr.__dataclass_fields__)
    assert fields == CORE_16_LOWERED_EXPR_FIELDS
    assert "prelude" not in fields
    assert "epilogue" not in fields
    assert "function_body" not in fields
    assert "invocation_scope" not in fields


def test_core_lowering_protocol_has_no_function_scope_hook() -> None:
    capability = inspect_core_function_scope_capability()
    assert capability.plugin_api_version == str(PLUGIN_API_VERSION)
    for name in PROPOSED_FUNCTION_SCOPE_HOOK_NAMES:
        assert name not in capability.protocol_hook_names
    assert capability.has_function_body_scope_hook is False
    assert capability.invocation_scope_optimization_active is False
    assert INVOCATION_SCOPE_OPTIMIZATION_ACTIVE is False
    assert "no deterministic" in capability.limitation or "LoweredExpr" in capability.limitation
    assert capability.proposal_id == PROPOSAL_ID
    assert capability.proposal_status == PROPOSAL_STATUS
    assert capability.lowered_expr_fields == CORE_16_LOWERED_EXPR_FIELDS


def test_plugin_object_does_not_advertise_scope_hooks() -> None:
    provider = plugin()
    for name in PROPOSED_FUNCTION_SCOPE_HOOK_NAMES:
        assert not hasattr(provider, name)


def test_proposal_payload_is_stack_raii_only() -> None:
    support = proposed_no_grad_function_body_support()
    assert support.rust_prelude == (
        "let _rxttorch_invocation_no_grad = tch::no_grad_guard();",
    )
    assert support.rust_epilogue == ()
    assert support.uses == ()
    assert support.helpers == ()
    joined = "\n".join(support.rust_prelude)
    assert "thread_local" not in joined
    assert "static " not in joined
    assert "lazy_static" not in joined
    assert "OnceLock" not in joined


def test_production_helpers_retain_per_operation_no_grad_guard() -> None:
    helpers = (
        linear_helper(),
        relu_helper(),
        tanh_helper(),
        sub_helper(),
        softmax_dim1_helper(),
        argmax_dim1_keepfalse_helper(),
    )
    for helper in helpers:
        assert "let _guard = tch::no_grad_guard();" in helper
        # Exactly one guard per helper body keeps the baseline unambiguous.
        assert helper.count("no_grad_guard") == 1
    summary = production_no_grad_baseline_summary()
    assert "per-operation" in summary or "each rextio-torch tch helper" in summary
    assert "invocation-scope optimization active=False" in summary


def test_proposal_document_exists_and_states_absence() -> None:
    path = ROOT / "docs" / "invocation-scope-proposal-0.1.3.md"
    text = path.read_text(encoding="utf-8")
    assert "proposal-only" in text
    assert "Absent" in text or "absent" in text
    assert "INVOCATION_SCOPE_OPTIMIZATION_ACTIVE" in text
    assert "per-operation" in text.lower()
    assert "thread locals" in text or "thread-local" in text
