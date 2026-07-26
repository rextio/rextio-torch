"""Core capability status for one RAII no-grad scope per native invocation.

Plugin API 1.6 (Core ``rextio`` 0.1.6) exposes only expression-level
:class:`~rextio.plugins.api.LoweredExpr` fields (``rust`` / ``uses`` /
``helpers``) plus module-level type helpers. There is no deterministic
function-body prelude or epilogue hook that plugins can use to install a
single stack-scoped ``tch::no_grad_guard()`` around an entire generated
native function.

Therefore production rextio-torch keeps the **per-operation**
``tch::no_grad_guard()`` baseline in every fallible helper. That RAII
guard is local to the helper call, drops on every Rust return path
(including ``?``-mapped ``TchError``), and cannot leak grad mode across
Python callbacks, exceptions, reentrancy, or nested generated calls.

This module records that limitation explicitly and describes the Core
extension that would enable a strictly plugin-local optimization with
identical exception and reentrancy semantics. It does **not** activate
any global, thread-local, tensor-lifetime, or hidden mutable-state
workaround.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Production path: per-op helpers own no_grad. This flag is the single
# source of truth that the invocation-scope optimization is inactive.
INVOCATION_SCOPE_OPTIMIZATION_ACTIVE: Final[bool] = False

# Closed set of LoweredExpr fields observed on Core plugin API 1.6.
CORE_16_LOWERED_EXPR_FIELDS: Final[frozenset[str]] = frozenset(
    {"rust", "uses", "helpers"}
)

# Hooks that would be required for a safe per-invocation scope. None of
# these exist on RextioLoweringPlugin / RextioPluginV2 in API 1.6.
PROPOSED_FUNCTION_SCOPE_HOOK_NAMES: Final[tuple[str, ...]] = (
    "function_body_support",
    "function_prelude",
    "function_epilogue",
    "invocation_scope",
    "native_function_scope",
)

PROPOSAL_ID: Final[str] = "rextio-torch/core-function-body-scope-v1"
PROPOSAL_STATUS: Final[str] = "proposal-only"
PROPOSAL_TARGET_PLUGIN_API: Final[str] = "1.7+"
PROPOSAL_TARGET_CORE: Final[str] = "rextio>=0.1.7 (not available; do not assume)"


@dataclass(frozen=True)
class FunctionBodySupportProposal:
    """Proposed Core payload for one generated native function body.

    Core would inject ``rust_prelude`` once after parameter binding and
    before the first lowered statement, and guarantee that every exit
    path (``return``, ``?``, panic-unwind if any) drops the RAII values
    in reverse construction order. Plugins must not rely on Drop running
    only on the happy path.

    ``rust_epilogue`` is optional and must be pure cleanup that is safe
    if the body already returned early; prefer RAII Drop over epilogue
    statements when possible.
    """

    rust_prelude: tuple[str, ...]
    rust_epilogue: tuple[str, ...] = ()
    uses: tuple[str, ...] = ()
    helpers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoreFunctionScopeCapability:
    """Deterministic inspection of whether Core can host invocation scopes."""

    plugin_api_version: str
    lowered_expr_fields: frozenset[str]
    protocol_hook_names: frozenset[str]
    has_function_body_scope_hook: bool
    invocation_scope_optimization_active: bool
    limitation: str
    proposal_id: str
    proposal_status: str


def _protocol_callable_names() -> frozenset[str]:
    """Return public callable names on the Core lowering protocol surface."""
    from rextio.plugins.api import RextioLoweringPlugin, RextioPluginV2

    names: set[str] = set()
    for protocol in (RextioPluginV2, RextioLoweringPlugin):
        for name, value in vars(protocol).items():
            if name.startswith("_"):
                continue
            if callable(value) or isinstance(value, property):
                names.add(name)
    return frozenset(names)


def _lowered_expr_field_names() -> frozenset[str]:
    """Return the frozen dataclass fields of Core ``LoweredExpr``."""
    from rextio.plugins.api import LoweredExpr

    return frozenset(LoweredExpr.__dataclass_fields__)


def inspect_core_function_scope_capability() -> CoreFunctionScopeCapability:
    """Inspect the live Core host for per-function scope hooks.

    Read-only and side-effect free aside from importing Core's plugin API.
    Always reports the production optimization as inactive on API 1.6.
    """
    from rextio.plugins.api import PLUGIN_API_VERSION

    fields = _lowered_expr_field_names()
    hooks = _protocol_callable_names()
    # Module-level helpers/uses on LoweredExpr are not a body prelude.
    has_scope_hook = any(name in hooks for name in PROPOSED_FUNCTION_SCOPE_HOOK_NAMES)
    # rextio-torch never claims the optimization is live without an explicit,
    # tested Core contract that preserves exception and reentrancy safety.
    has_function_body_scope_hook = has_scope_hook
    active = INVOCATION_SCOPE_OPTIMIZATION_ACTIVE

    if has_scope_hook:
        limitation = (
            "A function-scope-like name is present on the protocol, but "
            "rextio-torch 0.1.3 does not activate invocation-scope no_grad "
            "without a reviewed Core contract and identical exception/"
            "reentrancy semantics to per-operation guards."
        )
    else:
        limitation = (
            "Core plugin API 1.6 provides only LoweredExpr "
            "(rust/uses/helpers) and module-level type helpers; there is "
            "no deterministic per-generated-function or per-invocation "
            "Rust body prelude/epilogue hook. Production retains "
            "per-operation tch::no_grad_guard() in every helper."
        )

    return CoreFunctionScopeCapability(
        plugin_api_version=str(PLUGIN_API_VERSION),
        lowered_expr_fields=fields,
        protocol_hook_names=hooks,
        has_function_body_scope_hook=has_function_body_scope_hook,
        invocation_scope_optimization_active=active,
        limitation=limitation,
        proposal_id=PROPOSAL_ID,
        proposal_status=PROPOSAL_STATUS,
    )


def proposed_no_grad_function_body_support() -> FunctionBodySupportProposal:
    """Return the illustrative RAII scope plugins would emit under the proposal.

    This text is **not** injected into generated crates today. It exists so
    tests and docs can pin the desired semantics without pretending Core
    honors it.
    """
    return FunctionBodySupportProposal(
        rust_prelude=(
            # Stack-scoped RAII only. Drop restores prior grad mode on every
            # exit path of the generated native function body.
            "let _rxttorch_invocation_no_grad = tch::no_grad_guard();",
        ),
        rust_epilogue=(),
        uses=(),
        helpers=(),
    )


def production_no_grad_baseline_summary() -> str:
    """Return a short, stable description of the active production baseline."""
    return (
        "production baseline: each rextio-torch tch helper installs "
        "let _guard = tch::no_grad_guard(); inside the helper body; "
        f"invocation-scope optimization active={INVOCATION_SCOPE_OPTIMIZATION_ACTIVE}"
    )


__all__ = [
    "CORE_16_LOWERED_EXPR_FIELDS",
    "INVOCATION_SCOPE_OPTIMIZATION_ACTIVE",
    "PROPOSED_FUNCTION_SCOPE_HOOK_NAMES",
    "PROPOSAL_ID",
    "PROPOSAL_STATUS",
    "PROPOSAL_TARGET_CORE",
    "PROPOSAL_TARGET_PLUGIN_API",
    "CoreFunctionScopeCapability",
    "FunctionBodySupportProposal",
    "inspect_core_function_scope_capability",
    "production_no_grad_baseline_summary",
    "proposed_no_grad_function_body_support",
]
