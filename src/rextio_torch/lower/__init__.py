"""Lower router: dispatch claimed sites to Phase A lower modules."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.lower import activations, linear, reductions

__all__ = ["lower"]


def lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    """Emit the Rust expression for a previously claimed site.

    Independently revalidates authoritative claim metadata and fails closed
    with ``ValueError`` (not ``assert``) so guards survive ``python -O``.
    """
    for lane in (linear, activations, reductions):
        result = lane.try_lower(claimed, ctx)
        if result is not None:
            return result
    raise ValueError(
        f"rextio-torch cannot lower unclaimed site: {claimed.kind} {claimed.target!r}"
    )
