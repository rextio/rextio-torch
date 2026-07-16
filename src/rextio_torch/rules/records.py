"""Machine-readable rule records for rextio-torch Phase A."""

from rextio.plugins.api import RuleRecord, RuleScope

RULE_RECORDS: tuple[RuleRecord, ...] = (
    RuleRecord(
        id="rextio-torch/functional-linear-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "torch.nn.functional.linear(x, weight, bias) on float32 CPU rank-2 "
                "input/weight and float32 CPU rank-1 bias"
            ),
        ),
        constraint=(
            "Exactly three positional tensor operands: rank-2 float32 CPU input, "
            "rank-2 float32 CPU weight, and rank-1 float32 CPU bias. No keywords. "
            "Result is rank-2 float32 CPU. Executed under no-grad via fallible tch "
            "f_linear. Optional bias omission, other dtypes/devices/ranks, module "
            "forms, and in-place variants stay outside the Phase A surface."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-001",
        guidance=(
            "Annotate input and weight as rextio_torch.types.TensorF32Cpu2D and "
            "bias as TensorF32Cpu1D; call torch.nn.functional.linear with three "
            "positional arguments on CPU float32 tensors."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-relu-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="tensor.relu() on float32 CPU rank-2 tensors (method form)",
        ),
        constraint=(
            "Method call with no positional arguments and no keywords on a "
            "float32 CPU rank-2 receiver. Result preserves the receiver type. "
            "In-place relu_ and other ranks/dtypes stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-002",
        guidance=(
            "Call .relu() with no arguments on a TensorF32Cpu2D receiver; do not "
            "use in-place relu_."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-mean-dim1-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tensor.mean(dim=1, keepdim=False) on float32 CPU rank-2 tensors "
                "(method form with literal keywords)"
            ),
        ),
        constraint=(
            "Method call with no positional arguments and exactly the static "
            "keywords dim=<int 1> and keepdim=<bool False> on a float32 CPU "
            "rank-2 receiver. Result is float32 CPU rank-1. Dynamic dim/keepdim, "
            "other dimensions, keepdim=True, and whole-tensor mean stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-003",
        guidance=(
            "Write .mean(dim=1, keepdim=False) with literal keywords on a "
            "TensorF32Cpu2D receiver."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/unsupported-tensor-surface",
        provider="rextio-torch",
        scope=RuleScope(
            kind="type",
            pattern=(
                "tensor dtype/device/rank or call shape outside the Phase A "
                "float32 CPU rank-1/2 linear→ReLU→mean slice"
            ),
        ),
        constraint=(
            "Covered torch call sites whose operand or receiver types fall "
            "outside the registered float32 CPU rank-1/2 vocabulary, or whose "
            "shape is recognized but not lowerable, are rejected with guidance "
            "and stay on the Python fallback."
        ),
        outcome="fallback",
        diagnostic_code="RXTP-TORCH-010",
        guidance=(
            "Keep the Phase A slice on float32 CPU rank-2 input/weight, rank-1 "
            "bias, .relu(), and .mean(dim=1, keepdim=False); other dtypes, "
            "devices, ranks, and dynamic literals remain on the fallback."
        ),
        stability="experimental",
        verified=False,
    ),
)


def torch_rule_records() -> tuple[RuleRecord, ...]:
    """Return stable ordered rule records."""
    return RULE_RECORDS


__all__ = ["RULE_RECORDS", "torch_rule_records"]
