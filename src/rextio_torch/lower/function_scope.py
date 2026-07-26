"""Fail-closed access to Core API 1.7's provider-local guard fact."""

from __future__ import annotations


def function_scope_guard_active(ctx: object) -> bool:
    """Return true only for Core's explicit boolean activation signal.

    Legacy Core contexts and test doubles do not expose the attribute and
    therefore retain the per-operation ``no_grad`` helpers. Unexpected truthy
    values also fail closed to the guarded helper path.
    """
    return getattr(ctx, "function_scope_guard_active", False) is True


__all__ = ["function_scope_guard_active"]
