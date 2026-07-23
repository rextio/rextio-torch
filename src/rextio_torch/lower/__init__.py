"""Lower router: dispatch claimed sites to Alpha AOT lower modules."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.lower import activations, binops, classification, gelu, linear, reductions, unary

__all__ = ["lower"]


def lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    """Emit the Rust expression for a previously claimed site.

    Independently revalidates authoritative claim metadata and fails closed
    with ``ValueError`` (not ``assert``) so guards survive ``python -O``.
    """
    if getattr(ctx, "backend", "pyo3") != "pyo3":
        raise ValueError(
            "rextio-torch supports PyO3 host-extension lowering only; "
            "standalone artifacts are unsupported"
        )
    for lane in (linear, gelu, activations, unary, reductions, classification, binops):
        result = lane.try_lower(claimed, ctx)
        if result is not None:
            return result
    raise ValueError(
        f"rextio-torch cannot lower unclaimed site: {claimed.kind} {claimed.target!r}"
    )
