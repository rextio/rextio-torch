"""Core capability status for one RAII no-grad scope per native invocation.

Plugin API 1.7 (Core ``rextio`` 0.1.7) lets rextio-torch request one
stack-scoped ``tch::no_grad_guard()`` for an eligible generated PyO3 function.
Core installs it after tensor input conversion, drops it before tensor output
conversion, and relies on Rust RAII for early and error exits.

Functions containing an in-process Python fallback call (RXT075), standalone
backends, and type-only functions decline the whole-function guard. Their
lowering context remains inactive and the legacy helpers keep their
per-operation guards, so no no-grad state spans a Python callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Candidate production path: eligible native functions own one no-grad guard.
INVOCATION_SCOPE_OPTIMIZATION_ACTIVE: Final[bool] = True

# LoweredExpr remains expression-only in Core plugin API 1.7.
CORE_16_LOWERED_EXPR_FIELDS: Final[frozenset[str]] = frozenset(
    {"rust", "uses", "helpers"}
)
CORE_17_LOWERED_EXPR_FIELDS: Final[frozenset[str]] = CORE_16_LOWERED_EXPR_FIELDS

# Compatibility name retained for the former proposal inspection surface.
PROPOSED_FUNCTION_SCOPE_HOOK_NAMES: Final[tuple[str, ...]] = (
    "function_scope_guard",
)

PROPOSAL_ID: Final[str] = "rextio-torch/core-function-body-scope-v1"
PROPOSAL_STATUS: Final[str] = "implemented-candidate"
PROPOSAL_TARGET_PLUGIN_API: Final[str] = "1.7"
PROPOSAL_TARGET_CORE: Final[str] = "rextio>=0.1.7,<0.2"


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
    from rextio.plugins.api import (
        RextioFunctionScopeGuardPlugin,
        RextioLoweringPlugin,
        RextioPluginV2,
    )

    names: set[str] = set()
    for protocol in (
        RextioPluginV2,
        RextioLoweringPlugin,
        RextioFunctionScopeGuardPlugin,
    ):
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
    Reports active only when the live host exposes the reviewed API 1.7 hook.
    """
    from rextio.plugins.api import PLUGIN_API_VERSION

    fields = _lowered_expr_field_names()
    hooks = _protocol_callable_names()
    has_function_body_scope_hook = "function_scope_guard" in hooks
    version_parts = str(PLUGIN_API_VERSION).split(".")
    compatible = (
        len(version_parts) == 2
        and all(part.isdecimal() for part in version_parts)
        and int(version_parts[0]) == 1
        and int(version_parts[1]) >= 7
    )
    active = (
        INVOCATION_SCOPE_OPTIMIZATION_ACTIVE
        and compatible
        and has_function_body_scope_hook
    )

    if active:
        limitation = (
            "Core plugin API 1.7 provides the reviewed function_scope_guard "
            "contract. Eligible PyO3 functions use one no-grad RAII scope; "
            "RXT075, standalone, and type-only functions retain guarded helpers."
        )
    else:
        limitation = (
            "The live Core host does not expose the compatible plugin API 1.7 "
            "function_scope_guard contract; rextio-torch must not activate "
            "guardless function-scoped helpers."
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
    """Return the historical prelude shape now represented by the API 1.7 hook."""
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
        "production baseline: eligible native PyO3 functions install one "
        "tch::no_grad_guard(); RXT075/standalone/type-only paths retain "
        "per-operation guarded helpers; "
        f"invocation-scope optimization active={INVOCATION_SCOPE_OPTIMIZATION_ACTIVE}"
    )


__all__ = [
    "CORE_16_LOWERED_EXPR_FIELDS",
    "CORE_17_LOWERED_EXPR_FIELDS",
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
