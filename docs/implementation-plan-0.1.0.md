# rextio-torch 0.1.0 implementation plan

Status: public Alpha 0.1.0 released 2026-07-18

**Alpha AOT status:** Phase A vertical slice remains **real-Cargo certified**.
The broader fail-closed AOT surface (activations rank 1/2, matmul, elementwise
`+`, sum) is also certified in one serialized Cargo project.

**Host platforms:**

| Status | OS / arch | Contract |
| --- | --- | --- |
| **Certified** | macOS **arm64** | Real-Cargo Alpha evidence: CPython 3.11 / torch 2.11.0 / `LIBTORCH_USE_PYTORCH=1` (never `LIBTORCH_BYPASS_VERSION_CHECK`). |
| **Experimental** | Linux **x86_64**, Linux **AArch64** | Same pins; real hosted Cargo E2E (blocking x86_64, scheduled/manual AArch64). **Not** certified. |
| **Availability-gated** | macOS **x86_64** | No torch 2.11.0 CPython 3.11 x86_64 wheel; scheduled/manual CI verifies that absence. No support claim. |
| **Unsupported** | Linux/macOS **i686**, **ARMv7** | Static expected-unsupported tests; missing pinned runtimes and, on modern macOS, impossible runner targets. |
| **Deferred** | Windows | Unverified; no support claim in this cut. |

The plugin does **not** reject Linux solely by OS; unavailable or mismatched
toolchain/torch combinations must fail visibly under the pinned contract.

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

### Unreleased 0.1.2 bounded CPU follow-up

The next-version integration branch retains every 0.1.0 pin and boundary while
adding a fail-closed, real-Cargo-tested follow-up:

| Op | Added bounded contract |
| --- | --- |
| Activations | Exact `torch.relu/sigmoid/tanh(tensor)` with one positional float32 CPU rank-1/2 tensor |
| Elementwise `-` / true `/` | Same-rank 1d/2d or rank-2/rank-1 trailing broadcast in either operand order; binop only |
| Mean / sum | Receiver or exact `torch.mean/sum`; dim literal 0/1 once; keepdim omitted=False or named bool; only already-registered rank-1/2 outputs |
| Softmax | Receiver or exact `torch.softmax`; rank-1 dim0 or rank-2 dim0/1; no dtype/keepdim |
| Argmax | Exact int64 rank-1 outputs only: rank-2 dim0/1 keepdim=False or rank-1 dim0 keepdim=True |
| Functional linear without bias | Input/weight positional; bias omitted, positional literal None, or exact literal keyword `bias=None` |

Core API 1.3 does not represent tensor-valued keyword operands, so
`linear(..., bias=bias_tensor)` remains ordinary fallback. Positional
dimensions are independently checked through aligned `operand_literals` in
claim and lower, then omitted from the runtime tensor helper call. Positional
keepdim is not admitted; rank-2 argmax keepdim=True and rank-1 argmax
keepdim=False remain fallback because the exact int64 rank-2/rank-0 vocabulary
does not exist.

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
- Scalar operands and semantic-changing function aliases/options such as
  `torch.div(..., rounding_mode=...)` or `torch.add(..., alpha=...)`
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

- Package version is `0.1.0`, marked Development Status Alpha; the temporary
  `Private` upload-block classifier has been removed from the release.
- Use the public `rextio>=0.1.3,<0.2` dependency, not a core-next VCS pin.
- Annotated-tag, PyPI-artifact, and live no-cache-install evidence are recorded
  separately from this implementation contract.
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
- A declarative matrix covers Linux/macOS x86, x64, ARM32, and ARM64 without
  mistaking static negative tests for native support. Runtime-backed CI uses
  macOS ARM64 and Linux x86_64 on push/PR; Linux AArch64 is scheduled/manual;
  macOS x64 remains an artifact-availability gate.
- Package build plus `twine check` / `check-wheel-contents` succeeds before any
  release review (tools listed in the `dev` extra; gate not yet run for publish).

## Residual platform risks

- **Certified** real-Cargo evidence remains macOS arm64 only.
- Linux x86_64/AArch64 are **experimental** (real hosted E2E; not a
  certification host). Distro glibc/libstdc++, torch manylinux wheels, and cold
  `tch` builds are residual risks — not OS-level rejection by this plugin.
- Windows is **deferred** (unverified).
- `torch-sys` interpreter discovery depends on `PATH` / `VIRTUAL_ENV`.
- Cold native builds recompile `tch` against the active torch 2.11 install.
- CUDA/MPS remain out of scope.
- Core limitation: method claims require named or call-chain receivers, not
  bare BinOp receivers.

## Linux experimental smoke (not certification)

Maintainers may run the portable pin contract on Linux without weakening AOT
pins:

1. CPython 3.11 venv, `pip install -e '.[dev]'` (torch 2.11.0, rextio API 1.3).
2. `export LIBTORCH_USE_PYTORCH=1` and ensure `LIBTORCH_BYPASS_VERSION_CHECK` is
   unset.
3. Focused unit tests: `pytest -q tests --ignore=tests/e2e`.
4. Optional real-Cargo slice: `pytest -q tests/e2e -m needs_cargo` (or
   `./scripts/linux-smoke.sh --cargo`).

Passing smoke on Linux is engineering evidence only; it does not rewrite the
certified-host row until deliberately re-recorded.

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
