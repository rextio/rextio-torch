"""Machine-readable rule records for rextio-torch Alpha AOT."""

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
            "forms, and in-place variants stay outside the Alpha AOT surface."
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
            "In-place relu_ and other dtypes/devices stay unclaimed."
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
        id="rextio-torch/tensor-relu-f32-cpu-1d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="tensor.relu() on float32 CPU rank-1 tensors (method form)",
        ),
        constraint=(
            "Method call with no positional arguments and no keywords on a "
            "float32 CPU rank-1 receiver. Result preserves the receiver type."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-004",
        guidance="Call .relu() with no arguments on a TensorF32Cpu1D receiver.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-sigmoid-f32-cpu-1d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="tensor.sigmoid() on float32 CPU rank-1 tensors (method form)",
        ),
        constraint=(
            "Zero-argument method form on float32 CPU rank-1. Result preserves "
            "receiver type. In-place sigmoid_ is unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-005",
        guidance="Call .sigmoid() with no arguments on a TensorF32Cpu1D receiver.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-sigmoid-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="tensor.sigmoid() on float32 CPU rank-2 tensors (method form)",
        ),
        constraint=(
            "Zero-argument method form on float32 CPU rank-2. Result preserves "
            "receiver type. In-place sigmoid_ is unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-006",
        guidance="Call .sigmoid() with no arguments on a TensorF32Cpu2D receiver.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-tanh-f32-cpu-1d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="tensor.tanh() on float32 CPU rank-1 tensors (method form)",
        ),
        constraint=(
            "Zero-argument method form on float32 CPU rank-1. Result preserves "
            "receiver type. In-place tanh_ is unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-007",
        guidance="Call .tanh() with no arguments on a TensorF32Cpu1D receiver.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-tanh-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="tensor.tanh() on float32 CPU rank-2 tensors (method form)",
        ),
        constraint=(
            "Zero-argument method form on float32 CPU rank-2. Result preserves "
            "receiver type. In-place tanh_ is unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-008",
        guidance="Call .tanh() with no arguments on a TensorF32Cpu2D receiver.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-matmul-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern="rank-2 float32 CPU matmul via a @ b",
        ),
        constraint=(
            "Binary @ with two float32 CPU rank-2 operands. Result is rank-2 "
            "float32 CPU. Inner-dimension compatibility is not represented by "
            "the rank-only annotation vocabulary and is checked by tch at runtime. "
            "Rank-1, mixed ranks, and other dtypes/devices stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-009",
        guidance=(
            "Annotate both operands as TensorF32Cpu2D and write a @ b; ensure "
            "their concrete inner dimensions are compatible at runtime."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-matmul-call-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "rank-2 float32 CPU matmul via torch.matmul(a, b) or a.matmul(b)"
            ),
        ),
        constraint=(
            "Functional torch.matmul takes exactly two positional float32 CPU "
            "rank-2 operands; method .matmul takes a rank-2 receiver and one "
            "rank-2 positional operand. Neither form accepts keywords. Result is "
            "rank-2 float32 CPU. Concrete inner dimensions are checked by tch "
            "at runtime because annotations carry rank, not sizes."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-014",
        guidance=(
            "Annotate both tensors as TensorF32Cpu2D and call torch.matmul(a, b) "
            "or a.matmul(b) with runtime-compatible inner dimensions."
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
                "tensor dtype/device/rank or call shape outside the Alpha AOT "
                "float32 CPU rank-1/2 inference surface"
            ),
        ),
        constraint=(
            "Covered torch call/binop sites whose operand or receiver types fall "
            "outside the registered float32 CPU rank-1/2 vocabulary, or whose "
            "shape is recognized but not lowerable, are rejected with guidance "
            "and stay on the Python fallback."
        ),
        outcome="fallback",
        diagnostic_code="RXTP-TORCH-010",
        guidance=(
            "Keep the Alpha AOT surface on float32 CPU rank-1/2 tensors with "
            "claimed linear, activations, matmul, elementwise +/*, and literal "
            "mean/sum forms; other dtypes, devices, ranks, and dynamic literals "
            "remain on the fallback."
        ),
        stability="experimental",
        verified=False,
    ),
    RuleRecord(
        id="rextio-torch/tensor-add-f32-cpu-same-rank",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern="elementwise + on same-rank float32 CPU rank-1 or rank-2 tensors",
        ),
        constraint=(
            "Binary + with two float32 CPU tensors of equal rank (1 or 2). "
            "Result preserves the left operand type. Concrete sizes are not "
            "statically represented; tch checks same-shape or PyTorch-compatible "
            "same-rank broadcasting at runtime. Scalar operands stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-011",
        guidance=(
            "Write a + b with matching TensorF32Cpu1D or TensorF32Cpu2D annotations."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-add-f32-cpu-2d-1d-broadcast",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "elementwise + with rank-2 + rank-1 trailing bias broadcast "
                "(either order) on float32 CPU tensors"
            ),
        ),
        constraint=(
            "Binary + where one operand is float32 CPU rank-2 and the other is "
            "float32 CPU rank-1. Runtime tch applies PyTorch trailing-dimension "
            "broadcast rules and rejects incompatible concrete sizes. Result is "
            "rank-2 float32 CPU. Other rank combinations stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-012",
        guidance=(
            "Write rank2 + rank1 (or rank1 + rank2) with TensorF32Cpu2D / "
            "TensorF32Cpu1D annotations for trailing bias broadcast."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-mul-f32-cpu-same-rank",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern="elementwise * on same-rank float32 CPU rank-1 or rank-2 tensors",
        ),
        constraint=(
            "Binary * with two float32 CPU tensors of equal rank (1 or 2). "
            "Result preserves the left operand type. Concrete sizes are not "
            "statically represented; tch checks same-shape or PyTorch-compatible "
            "same-rank broadcasting at runtime. Scalar operands stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-017",
        guidance=(
            "Write a * b with matching TensorF32Cpu1D or TensorF32Cpu2D annotations."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-mul-f32-cpu-2d-1d-broadcast",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "elementwise * with rank-2 * rank-1 trailing bias broadcast "
                "(either order) on float32 CPU tensors"
            ),
        ),
        constraint=(
            "Binary * where one operand is float32 CPU rank-2 and the other is "
            "float32 CPU rank-1. Runtime tch applies PyTorch trailing-dimension "
            "broadcast rules and rejects incompatible concrete sizes. Result is "
            "rank-2 float32 CPU. Other rank combinations stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-018",
        guidance=(
            "Write rank2 * rank1 (or rank1 * rank2) with TensorF32Cpu2D / "
            "TensorF32Cpu1D annotations for trailing bias broadcast."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-sum-dim1-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tensor.sum(dim=1, keepdim=False) on float32 CPU rank-2 tensors "
                "(method form with literal keywords)"
            ),
        ),
        constraint=(
            "Same literal keyword contract as mean: dim=1, keepdim=False on a "
            "float32 CPU rank-2 receiver. Result is float32 CPU rank-1. Dynamic "
            "or other dimensions stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-013",
        guidance=(
            "Write .sum(dim=1, keepdim=False) with literal keywords on a "
            "TensorF32Cpu2D receiver."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-softmax-dim1-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="tensor.softmax(dim=1) on float32 CPU rank-2 logits (method form)",
        ),
        constraint=(
            "Method call with no positional arguments and exactly the literal keyword "
            "dim=<int 1> on a float32 CPU rank-2 receiver. No dtype keyword or "
            "functional spelling is claimed. Result remains float32 CPU rank-2."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-015",
        guidance="Write .softmax(dim=1) with a literal keyword on TensorF32Cpu2D logits.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-argmax-dim1-keepfalse-i64-cpu-1d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tensor.argmax(dim=1, keepdim=False) on float32 CPU rank-2 logits "
                "(method form)"
            ),
        ),
        constraint=(
            "Method call with no positional arguments and exactly literal dim=<int 1> "
            "and keepdim=<bool False> on a float32 CPU rank-2 receiver. The result "
            "is checked and materialized as an int64 CPU rank-1 tensor. Dynamic values, "
            "keepdim=True, dtype overrides, and functional spellings stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-016",
        guidance=(
            "Write .argmax(dim=1, keepdim=False) after rank-2 float32 CPU logits; "
            "annotate the function result as TensorI64Cpu1D."
        ),
        stability="experimental",
        verified=True,
    ),
)


def torch_rule_records() -> tuple[RuleRecord, ...]:
    """Return stable ordered rule records."""
    return RULE_RECORDS


__all__ = ["RULE_RECORDS", "torch_rule_records"]
