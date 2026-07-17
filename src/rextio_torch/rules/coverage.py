"""Declared torch package and symbol coverage for Alpha AOT."""

from rextio.plugins.api import CoverageDecl

# ``symbols`` is descriptive (capability manifest). Method forms and binops
# are claimed via plugin API 1.3 receiver/operand type matching rather than
# module-qualified symbols alone; functional linear and torch.matmul are the
# explicit module-call forms.
COVERAGE = CoverageDecl(
    packages=("torch",),
    modules=("torch", "torch.nn", "torch.nn.functional"),
    symbols=(
        "torch.nn.functional.linear",
        "torch.matmul",
        "torch.Tensor.relu",
        "torch.Tensor.sigmoid",
        "torch.Tensor.tanh",
        "torch.Tensor.mean",
        "torch.Tensor.sum",
        "torch.Tensor.matmul",
    ),
)

__all__ = ["COVERAGE"]
