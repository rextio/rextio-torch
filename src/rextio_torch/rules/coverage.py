"""Declared torch package and symbol coverage for Alpha AOT."""

from rextio.plugins.api import CoverageDecl

# ``symbols`` is descriptive (capability manifest). Method forms and binops
# are claimed via plugin API 1.3+ receiver/operand type matching rather than
# module-qualified symbols alone; functional linear and torch.matmul are the
# explicit module-call forms.
COVERAGE = CoverageDecl(
    packages=("torch",),
    modules=("torch", "torch.nn", "torch.nn.functional"),
    symbols=(
        "torch.nn.functional.linear",
        "torch.nn.functional.gelu",
        "torch.nn.functional.relu",
        "torch.nn.functional.softmax",
        "torch.matmul",
        "torch.relu",
        "torch.sigmoid",
        "torch.tanh",
        "torch.mean",
        "torch.sum",
        "torch.softmax",
        "torch.argmax",
        "torch.add",
        "torch.sub",
        "torch.mul",
        "torch.div",
        "torch.abs",
        "torch.neg",
        "torch.negative",
        "torch.square",
        "torch.exp",
        "torch.log",
        "torch.sqrt",
        "torch.Tensor.relu",
        "torch.Tensor.sigmoid",
        "torch.Tensor.tanh",
        "torch.Tensor.mean",
        "torch.Tensor.sum",
        "torch.Tensor.softmax",
        "torch.Tensor.argmax",
        "torch.Tensor.matmul",
        "torch.Tensor.abs",
        "torch.Tensor.neg",
        "torch.Tensor.negative",
        "torch.Tensor.square",
        "torch.Tensor.exp",
        "torch.Tensor.log",
        "torch.Tensor.sqrt",
    ),
)

__all__ = ["COVERAGE"]
