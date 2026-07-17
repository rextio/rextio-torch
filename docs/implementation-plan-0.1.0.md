# rextio-torch 0.1.0 implementation plan

Status: private, unreleased incubation

**Alpha AOT status:** Phase A vertical slice remains **real-Cargo certified**.
The broader fail-closed AOT surface (activations rank 1/2, matmul, elementwise
`+`, sum) is also certified in one serialized Cargo project. Tested environment:
**macOS arm64 / CPython 3.11 / torch 2.11.0** with `LIBTORCH_USE_PYTORCH=1`
(never `LIBTORCH_BYPASS_VERSION_CHECK`).

**Performance:** Phase A/B product-route benchmarks are **historical context
only** (see `benchmarks/results/` and `benchmarks/results_phase_b/`). They are
not a release gate for Alpha AOT usefulness.

## Product definition

`rextio-torch` is a Rextio plugin that lowers a statically proven subset of
Python PyTorch inference code to Rust expressions backed by `tch`.

It is not a whole-project PyTorch translator. Unsupported or unproven sites
must remain on the ordinary Python fallback path. The first release is CPU
inference only.

## Compatibility baseline

- Python: CPython 3.11 (`requires-python = ">=3.11,<3.12"`)
- Rextio: `>=0.1.3,<0.2`
- Plugin API: `1.3`
- PyTorch: `torch==2.11.0`
- Rust binding: `tch==0.24.0` with the `python-extension` feature
- Device: CPU only
- Mode: inference / no-grad only

The published `tch` 0.24.0 release targets PyTorch/libtorch 2.11.0 exactly.
`LIBTORCH_BYPASS_VERSION_CHECK` is not an accepted build or release path.
Builds using the Python PyTorch installation set `LIBTORCH_USE_PYTORCH=1` and
must use the same active Python 3.11 environment as Rextio/PyO3.

## Boundary and ownership contract

The exported Python boundary uses materialized plugin types whose native Rust
representation is a plugin-owned `RxtTorchTensor(tch::Tensor)`.

- Python `torch.Tensor` input is unpacked through tch's Python-extension
  bridge.
- Rust `tch::Tensor` output is wrapped through the same bridge.
- This creates another ref-counted tensor handle and does not copy tensor
  storage.
- `RxtTorchTensor::clone()` uses `Tensor::shallow_clone()`.
- In-place operations are excluded because shallow clones alias storage.
- A materialized boundary type remains a native Rust value between claimed
  expressions; no intermediate Python materialization is allowed.
- Truly resident API 1.3 types (`conversion=None`) cannot cross an exported
  PyO3 boundary and are not required for the first vertical slice.

Boundary extraction validates the promised CPU device, dtype, and rank at
runtime and returns a Python exception on a mismatch (stable `rextio-torch:`
messages). Plugin discovery and annotation imports must not import `torch`.

## Phase A: first working vertical slice (done, certified)

End-to-end route for float32 CPU tensors:

```python
def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    return torch.nn.functional.linear(x, weight, bias).relu().mean(
        dim=1, keepdim=False
    )
```

The generated Rust path:

1. unpacks all three Python tensors without copying their storage;
2. execute functional linear, ReLU, and the literal-dimension reduction using
   fallible `tch` APIs under no-grad semantics;
3. retain the intermediate tensors entirely in Rust;
4. wrap the final tensor back into a Python `torch.Tensor`;
5. preserve input tensors and produce no autograd graph.

Real-Cargo certification (single serialized build) covers native route
evidence, numerical equivalence to eager PyTorch, CPU/float32/rank-1 output,
no-grad output when inputs request gradients, input non-mutation, returned
output lifetime after inputs are released, and fail-closed boundary rejects
for runtime dtype/rank violations.

Native Phase A and expanded Alpha rule records are marked `verified=True` after
the serialized certification project compiles and executes every rule family.

## Alpha AOT surface (implemented)

Same fail-closed claim → independent lower revalidation → fallible no-grad
helpers:

| Op | Metadata contract |
| --- | --- |
| `.relu()` / `.sigmoid()` / `.tanh()` | Zero-arg method; rank-1 or rank-2; result = receiver |
| Rank-2 matmul | `@` / `torch.matmul` / `.matmul`; two rank-2 operands; inner sizes runtime-checked |
| Elementwise `+` | Same-rank 1d/2d or rank-2 + rank-1; concrete broadcast sizes runtime-checked |
| `.mean` / `.sum` | `dim=1`, `keepdim=False` literals only; rank-2 → rank-1 |
| Functional linear | Unchanged Phase A three-operand form |

### Control-flow vertical slice

Python `for` / `if` around matmul, bias `+`, relu/sigmoid, and mean lower to
Rust control flow around tch helpers when their conditions are Rextio-native
scalar `int` / `bool` expressions. Tensor comparisons and tensor-data-dependent
conditions are not claimable. Because core does not claim method calls whose
receiver is a bare BinOp, kernels use named temps:

```python
even = hidden @ weight + bias
hidden = even.relu()
```

Use distinct temp names per if-arm when both branches need intermediates (core
scopes Rust `let` bindings per arm).

### Intentionally not claimed (without weakening guards)

- Literal transpose / reshape / view (view/alias ambiguity with shallow clone)
- Elementwise `-`, `*`, `/` and scalar operands (not yet proven in this cut)
- Whole-tensor mean/sum, dynamic dim/keepdim
- In-place methods and operators

Every claim is determined from authoritative API 1.3 type, receiver, callable,
schema, and literal metadata. Lowering independently revalidates metadata and
fails closed with explicit exceptions rather than assertions. Unsupported forms
remain fallback or rejected — never falsely claimed.

The annotation vocabulary proves dtype, device, and rank, not concrete tensor
sizes. The original libtorch runtime therefore remains authoritative for inner
matmul dimensions and concrete broadcasting compatibility; incompatible sizes
raise through the fallible tch path.

## Explicit exclusions

- CUDA, MPS, or other devices
- autograd, training, backward, optimizers, and parameter mutation
- arbitrary `nn.Module` conversion or execution
- dynamic dtype, device, rank, reduction dimensions, or reshape rank
- tensor comparisons and tensor-data-dependent Python conditions
- custom operators
- in-place methods and operators
- ambiguous alias/view behavior
- unsupported broadcasting
- version-check bypasses

## Repository and release safeguards

- Package version is `0.1.0`, marked unreleased.
- The private-incubator package metadata includes
  `Private :: Do Not Upload`.
- Use the public `rextio>=0.1.3,<0.2` dependency, not a core-next VCS pin.
- Do not tag, publish to PyPI, or change GitHub visibility during incubation.
- Do not add or modify a project-local `AGENTS.md` without owner direction.

## Acceptance checks

- Plugin entry point loads against Rextio plugin API 1.3.
- Coverage, rules, diagnostic namespace, exact crate pin and feature, type
  vocabulary, and import-minimal behavior have focused unit tests.
- Unsupported dtype/device/rank/dynamic literal/in-place cases are not claimed
  or are explicitly rejected according to the documented authority.
- Lowering has focused source/codegen tests and does not rely on `assert`.
- Real-Cargo tests prove Phase A and Alpha control-flow routes when the pinned
  local environment is available.
- Package build plus `twine check` / `check-wheel-contents` succeeds before any
  release review (tools listed in the `dev` extra; gate not yet run for publish).

## Residual platform risks

- Certification proven on macOS arm64 only so far.
- `torch-sys` interpreter discovery depends on `PATH` / `VIRTUAL_ENV`.
- Cold native builds recompile `tch` against the active torch 2.11 install.
- Other OS/arch, CUDA/MPS remain out of scope.
- Core limitation: method claims require named or call-chain receivers, not
  bare BinOp receivers.

## Historical benchmark context (not a gate)

Benchmark only the generated Rextio wrapper, not a standalone tch prototype.
Phase A and Phase B preregistered matrices recorded boundary-inclusive
native vs eager-fallback latency under fixed threads/warmups. Retained
artifacts:

- `benchmarks/results/latest.json` + `report.md` (Phase A)
- `benchmarks/results_phase_b/latest.json` + `report.md` (Phase B)

Those numbers document measured product-route latency on the recorded machine.
They do **not** gate shipping the Alpha AOT surface; the product goal for this
cut is a tightly pinned, useful native-AOT slice with fail-closed correctness.
