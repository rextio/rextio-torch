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
        id="rextio-torch/functional-linear-none-f32-cpu-2d",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "torch.nn.functional.linear(input, weight) with bias omitted, "
                "literal positional None, or exact literal keyword bias=None"
            ),
        ),
        constraint=(
            "Exact canonical functional.linear target with float32 CPU rank-2 "
            "input and weight. Bias is absent or statically proved literal None; "
            "result is rank-2 float32 CPU under no-grad. Tensor-valued keyword "
            "bias/input/weight cannot be represented by Core plugin API 1.3 and "
            "stays fallback. Other keywords and module forms are excluded."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-023",
        guidance=(
            "Pass input/weight positionally. Omit bias, pass positional literal "
            "None, or use the exact literal keyword bias=None."
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
            "Mixed rank-2/rank-1 forms use the sibling mixed-rank rule; rank-1 × "
            "rank-1 and other dtypes/devices stay unclaimed."
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
        id="rextio-torch/tensor-matmul-f32-cpu-mixed-rank",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "mixed-rank float32 CPU matrix/vector matmul via rank-2 @ rank-1 "
                "or rank-1 @ rank-2"
            ),
        ),
        constraint=(
            "Binary @ with one float32 CPU rank-2 operand and one float32 CPU "
            "rank-1 operand, in either order. Result is rank-1 float32 CPU. "
            "Concrete inner dimensions remain a fallible tch runtime contract. "
            "Rank-1 @ rank-1 is excluded because it produces an unregistered "
            "rank-0 result; other dtypes/devices/ranks remain fallback."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-035",
        guidance=(
            "Use TensorF32Cpu2D @ TensorF32Cpu1D or the reverse order and ensure "
            "their concrete inner dimensions are compatible at runtime."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-matmul-call-f32-cpu-mixed-rank",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "mixed-rank float32 CPU matrix/vector matmul via "
                "torch.matmul(a, b) or a.matmul(b)"
            ),
        ),
        constraint=(
            "Exact functional torch.matmul with two positional operands, or "
            "zero-keyword method .matmul with one positional operand, where one "
            "tensor is float32 CPU rank-2 and the other is float32 CPU rank-1. "
            "Either order returns float32 CPU rank-1. Concrete inner dimensions "
            "are checked by fallible tch f_matmul at runtime. Rank-1 × rank-1, "
            "keywords, and other dtypes/devices/ranks remain fallback."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-036",
        guidance=(
            "Call torch.matmul(matrix, vector), torch.matmul(vector, matrix), "
            "matrix.matmul(vector), or vector.matmul(matrix) without keywords."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/unary-abs-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="exact torch.abs(tensor) or zero-argument tensor.abs()",
        ),
        constraint=(
            "One positional float32 CPU rank-1/rank-2 tensor for torch.abs, or "
            "a zero-argument .abs() receiver of the same types; no keywords. "
            "Result preserves type and executes through fallible tch f_abs under "
            "no-grad. Scalar, out, in-place, and alternate aliases stay fallback."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-037",
        guidance="Use exact torch.abs(tensor) or tensor.abs() without keywords.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/unary-neg-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="exact torch.neg(tensor) or zero-argument tensor.neg()",
        ),
        constraint=(
            "Exact neg spelling on one float32 CPU rank-1/rank-2 tensor, with "
            "no keywords or method arguments. Result preserves type and uses "
            "fallible tch f_neg under no-grad; in-place/scalar/out forms remain fallback."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-038",
        guidance="Use exact torch.neg(tensor) or tensor.neg() without keywords.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/unary-negative-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="exact torch.negative(tensor) or zero-argument tensor.negative()",
        ),
        constraint=(
            "Exact negative spelling on one float32 CPU rank-1/rank-2 tensor, "
            "with no keywords or method arguments. Result preserves type and "
            "uses the distinct fallible tch f_negative path under no-grad."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-039",
        guidance=(
            "Use exact torch.negative(tensor) or tensor.negative() without keywords."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/unary-square-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="exact torch.square(tensor) or zero-argument tensor.square()",
        ),
        constraint=(
            "Exact square spelling on one float32 CPU rank-1/rank-2 tensor, with "
            "no keywords or method arguments. Result preserves type and uses "
            "fallible tch f_square under no-grad."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-040",
        guidance="Use exact torch.square(tensor) or tensor.square() without keywords.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/unary-exp-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="exact torch.exp(tensor) or zero-argument tensor.exp()",
        ),
        constraint=(
            "Exact exp spelling on one float32 CPU rank-1/rank-2 tensor, with no "
            "keywords or method arguments. Result preserves type and uses fallible "
            "tch f_exp under no-grad; IEEE overflow/underflow remains backend behavior."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-041",
        guidance="Use exact torch.exp(tensor) or tensor.exp() without keywords.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/unary-log-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="exact torch.log(tensor) or zero-argument tensor.log()",
        ),
        constraint=(
            "Exact log spelling on one float32 CPU rank-1/rank-2 tensor, with no "
            "keywords or method arguments. Result preserves type and uses fallible "
            "tch f_log under no-grad; domain NaN and zero-to-infinity behavior is "
            "the pinned backend contract rather than an operation error."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-042",
        guidance="Use exact torch.log(tensor) or tensor.log() without keywords.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/unary-sqrt-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="exact torch.sqrt(tensor) or zero-argument tensor.sqrt()",
        ),
        constraint=(
            "Exact sqrt spelling on one float32 CPU rank-1/rank-2 tensor, with no "
            "keywords or method arguments. Result preserves type and uses fallible "
            "tch f_sqrt under no-grad; negative-domain NaN and signed-zero behavior "
            "follow the pinned backend."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-043",
        guidance="Use exact torch.sqrt(tensor) or tensor.sqrt() without keywords.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/functional-gelu-none-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "exact torch.nn.functional.gelu(tensor) with approximate omitted "
                "or exact literal keyword approximate='none'"
            ),
        ),
        constraint=(
            "One positional float32 CPU rank-1/rank-2 tensor and either no "
            "keywords or exactly the static string literal approximate='none'. "
            "Both forms lower to fixed fallible tch f_gelu(\"none\") under no-grad; "
            "the source keyword is never interpolated. approximate='tanh', "
            "dynamic/other keywords, positional approximate, modules, methods, "
            "and other dtypes/devices/ranks remain fallback. Backward/training "
            "use stays out of scope; the helper always returns a no-grad result."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-044",
        guidance=(
            "Use torch.nn.functional.gelu(tensor) or pass only the exact literal "
            "keyword approximate='none' in inference/no-grad code."
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
        id="rextio-torch/tensor-sub-f32-cpu-same-rank",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern="elementwise - on same-rank float32 CPU rank-1 or rank-2 tensors",
        ),
        constraint=(
            "Binary - with two float32 CPU tensors of equal rank (1 or 2). "
            "Result preserves the left operand type. Concrete sizes are not "
            "statically represented; tch checks same-shape or PyTorch-compatible "
            "same-rank broadcasting at runtime. Scalar operands stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-019",
        guidance=(
            "Write a - b with matching TensorF32Cpu1D or TensorF32Cpu2D annotations."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-sub-f32-cpu-2d-1d-broadcast",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "elementwise - with rank-2/rank-1 trailing broadcast "
                "(either operand order) on float32 CPU tensors"
            ),
        ),
        constraint=(
            "Binary - where one operand is float32 CPU rank-2 and the other is "
            "float32 CPU rank-1. Operand order is preserved because subtraction "
            "is not commutative. Runtime tch applies PyTorch trailing-dimension "
            "broadcast rules and rejects incompatible concrete sizes. Result is "
            "rank-2 float32 CPU."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-020",
        guidance=(
            "Write rank2 - rank1 or rank1 - rank2 with TensorF32Cpu2D / "
            "TensorF32Cpu1D annotations; concrete trailing sizes must broadcast."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-div-f32-cpu-same-rank",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern="true division / on same-rank float32 CPU rank-1 or rank-2 tensors",
        ),
        constraint=(
            "Binary / with two float32 CPU tensors of equal rank (1 or 2). "
            "Result preserves the left operand type. tch performs PyTorch true "
            "division, including IEEE-754 zero/NaN/Inf behavior. Scalar operands "
            "and non-float32 tensors stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-021",
        guidance=(
            "Write a / b with matching TensorF32Cpu1D or TensorF32Cpu2D annotations."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/tensor-div-f32-cpu-2d-1d-broadcast",
        provider="rextio-torch",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "true division / with rank-2/rank-1 trailing broadcast "
                "(either operand order) on float32 CPU tensors"
            ),
        ),
        constraint=(
            "Binary / where one operand is float32 CPU rank-2 and the other is "
            "float32 CPU rank-1. Operand order is preserved because division is "
            "not commutative. Runtime tch applies PyTorch trailing-dimension "
            "broadcast rules and rejects incompatible concrete sizes. Result is "
            "rank-2 float32 CPU."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-022",
        guidance=(
            "Write rank2 / rank1 or rank1 / rank2 with TensorF32Cpu2D / "
            "TensorF32Cpu1D annotations; concrete trailing sizes must broadcast."
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
        id="rextio-torch/function-relu-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="torch.relu(tensor) on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exact target torch.relu with one positional tensor and no keywords. "
            "Result preserves the float32 CPU rank-1/rank-2 input type. Other "
            "functional namespaces and in-place forms stay unclaimed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-024",
        guidance="Call torch.relu(tensor) with one annotated float32 CPU tensor.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/functional-relu-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="torch.nn.functional.relu(tensor, inplace=False) on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exact torch.nn.functional.relu with one positional tensor and inplace "
            "omitted or exactly the literal False. True, dynamic, duplicate, and all "
            "other options remain fail-closed."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-050",
        guidance="Call F.relu(tensor) or F.relu(tensor, inplace=False).",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/function-sigmoid-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="torch.sigmoid(tensor) on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exact target torch.sigmoid with one positional tensor and no keywords. "
            "Result preserves the registered input type; other call shapes stay fallback."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-025",
        guidance="Call torch.sigmoid(tensor) with one annotated float32 CPU tensor.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/function-tanh-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="torch.tanh(tensor) on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exact target torch.tanh with one positional tensor and no keywords. "
            "Result preserves the registered input type; other call shapes stay fallback."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-026",
        guidance="Call torch.tanh(tensor) with one annotated float32 CPU tensor.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/mean-static-dim-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "torch.mean(tensor, dim) or tensor.mean(dim) with static dim/keepdim "
                "and an already-representable float32 CPU rank-1/rank-2 result"
            ),
        ),
        constraint=(
            "Exact torch.mean functional form or receiver method. dim is literal 0/1 "
            "and appears once, positionally or by keyword. keepdim is omitted (False) "
            "or a named bool literal; positional keepdim and dtype/out options are "
            "excluded. Rank-2 supports dim 0/1 with either keepdim value. Rank-1 "
            "supports only dim=0, keepdim=True because scalar rank-0 is unregistered."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-027",
        guidance=(
            "Use a literal dim once and keep the result inside TensorF32Cpu1D/2D; "
            "pass keepdim only as a named bool literal."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/sum-static-dim-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "torch.sum(tensor, dim) or tensor.sum(dim) with static dim/keepdim "
                "and an already-representable float32 CPU rank-1/rank-2 result"
            ),
        ),
        constraint=(
            "Same bounded form/type matrix as static mean: literal dim 0/1 once, "
            "optional named bool keepdim (default False), no positional keepdim or "
            "dtype/out. Rank-1 keepdim=False remains fallback because rank-0 is absent."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-028",
        guidance=(
            "Use a literal dim once and keep the result inside TensorF32Cpu1D/2D; "
            "pass keepdim only as a named bool literal."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/softmax-static-dim-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "torch.softmax(tensor, dim) or tensor.softmax(dim) with literal "
                "dim 0/1 on representable float32 CPU rank-1/rank-2 tensors"
            ),
        ),
        constraint=(
            "Exact torch.softmax functional form or receiver method. dim appears "
            "once as a literal positional/keyword value. Rank-1 accepts dim=0; "
            "rank-2 accepts dim=0/1. dtype and every other option are excluded; "
            "softmax has no keepdim parameter."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-029",
        guidance="Use torch.softmax(tensor, dim=<literal>) without dtype or keepdim.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/functional-softmax-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="torch.nn.functional.softmax(tensor, dim) on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exact torch.nn.functional.softmax with one positional tensor and a literal "
            "dim once. Rank-1 accepts dim=0; rank-2 accepts dim=0/1. dtype is "
            "omitted or exactly the literal None; _stacklevel and every other option "
            "are excluded."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-051",
        guidance="Call F.softmax(tensor, dim=<literal>) with dtype omitted or None.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/argmax-static-dim-i64-cpu-rank1",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern=(
                "torch.argmax(tensor, dim) or tensor.argmax(dim) with a literal "
                "dimension and an exact int64 CPU rank-1 result"
            ),
        ),
        constraint=(
            "dim appears once as literal 0/1; keepdim is omitted (False) or a "
            "named bool literal. Rank-2 dim 0/1 is accepted only with keepdim=False. "
            "Rank-1 dim=0 is accepted only with keepdim=True. Rank-2 keepdim=True "
            "and rank-1 keepdim=False remain fallback because no exact registered "
            "int64 rank-2/rank-0 result type exists."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-030",
        guidance=(
            "Keep argmax output at TensorI64Cpu1D: rank-2 with keepdim=False or "
            "rank-1 dim=0 with keepdim=True."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/function-add-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="torch.add(a, b) on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exact target torch.add with exactly two positional tensor operands and no "
            "keywords. The result follows the certified operator + matrix: same-rank "
            "rank-1/rank-2 or rank-2/rank-1 trailing broadcast. alpha, out, scalar, "
            "keyword, and other overloads stay fallback."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-031",
        guidance="Call torch.add(a, b) with two positional TensorF32Cpu1D/2D values.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/function-sub-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="torch.sub(a, b) on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exact target torch.sub with two positional tensors and no keywords. "
            "Operand order is preserved across the existing same-rank and rank-2/rank-1 "
            "broadcast matrix. alpha, out, scalar, keyword, and alias forms are excluded."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-032",
        guidance="Call torch.sub(a, b) with two positional TensorF32Cpu1D/2D values.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/function-mul-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="torch.mul(a, b) on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exact target torch.mul with two positional tensors and no keywords, using "
            "the certified operator * same-rank and rank-2/rank-1 broadcast matrix. "
            "out, scalar, keyword, and other aliases stay fallback."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-033",
        guidance="Call torch.mul(a, b) with two positional TensorF32Cpu1D/2D values.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-torch/function-div-f32-cpu-rank1-2",
        provider="rextio-torch",
        scope=RuleScope(
            kind="call",
            pattern="torch.div(a, b) on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exact target torch.div with two positional tensors and no keywords. "
            "Operand order is preserved across the certified true-division matrix. "
            "rounding_mode, out, scalar, keyword, and alias forms stay fallback."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-034",
        guidance="Call torch.div(a, b) with two positional TensorF32Cpu1D/2D values.",
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
    RuleRecord(
        id="rextio-torch/cuda0-matmul-f32-2d",
        provider="rextio-torch",
        scope=RuleScope(kind="binop", pattern="rank-2 CUDA device-0 float32 @"),
        constraint=(
            "Exactly TensorF32Cuda0_2D @ TensorF32Cuda0_2D under the selected "
            "rextio-device-cuda cuda-libtorch-linux-x86_64 authorization."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-045",
        guidance="Use two float32 rank-2 tensors already resident on cuda:0.",
        stability="experimental",
        verified=False,
    ),
    RuleRecord(
        id="rextio-torch/cuda0-bias-add-f32-2d-1d",
        provider="rextio-torch",
        scope=RuleScope(kind="binop", pattern="rank-2 + rank-1 CUDA device-0 bias add"),
        constraint=(
            "Exactly TensorF32Cuda0_2D + TensorF32Cuda0_1D. Reverse order, "
            "same-rank addition, transfers, and mixed devices are excluded."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-046",
        guidance="Keep the cuda:0 rank-2 matrix first and cuda:0 rank-1 bias second.",
        stability="experimental",
        verified=False,
    ),
    RuleRecord(
        id="rextio-torch/cuda0-relu-f32-2d",
        provider="rextio-torch",
        scope=RuleScope(kind="call", pattern="zero-argument CUDA device-0 rank-2 .relu()"),
        constraint="Method form only on TensorF32Cuda0_2D; functional and in-place forms excluded.",
        outcome="native",
        diagnostic_code="RXTP-TORCH-047",
        guidance="Call .relu() on an already-resident cuda:0 rank-2 tensor.",
        stability="experimental",
        verified=False,
    ),
    RuleRecord(
        id="rextio-torch/cuda0-mean-dim1-f32-2d",
        provider="rextio-torch",
        scope=RuleScope(kind="call", pattern="CUDA device-0 rank-2 .mean(dim=1)"),
        constraint=(
            "Method form on TensorF32Cuda0_2D with literal dim=1 and omitted or "
            "literal False keepdim, producing TensorF32Cuda0_1D."
        ),
        outcome="native",
        diagnostic_code="RXTP-TORCH-048",
        guidance="Call .mean(dim=1) with a literal dimension on the cuda:0 rank-2 value.",
        stability="experimental",
        verified=False,
    ),
)


def torch_rule_records() -> tuple[RuleRecord, ...]:
    """Return stable ordered rule records."""
    return RULE_RECORDS


__all__ = ["RULE_RECORDS", "torch_rule_records"]
