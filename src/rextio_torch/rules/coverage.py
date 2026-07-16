"""Declared torch package and symbol coverage for Phase A."""

from rextio.plugins.api import CoverageDecl

# ``symbols`` is descriptive (capability manifest). Method forms (``.relu``,
# ``.mean``) are claimed via plugin API 1.3 receiver-type matching rather than
# module-qualified symbols; the functional linear spelling is the explicit
# module-call form for the Phase A slice.
COVERAGE = CoverageDecl(
    packages=("torch",),
    modules=("torch", "torch.nn", "torch.nn.functional"),
    symbols=(
        "torch.nn.functional.linear",
        "torch.Tensor.relu",
        "torch.Tensor.mean",
    ),
)

__all__ = ["COVERAGE"]
