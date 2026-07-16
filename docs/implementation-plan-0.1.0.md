# rextio-torch 0.1.0 implementation plan

Status: private, unreleased incubation

**Phase A status:** implemented and **real-Cargo certified** on the tested
**macOS arm64 / CPython 3.11 / torch 2.11.0** environment with
`LIBTORCH_USE_PYTORCH=1` (never `LIBTORCH_BYPASS_VERSION_CHECK`). Broader 0.1.0
stretch surface remains open.

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

The plugin may expose equivalent function spellings where Rextio API 1.3
metadata can prove the same operation. Arbitrary `nn.Module`, `.eval()`, and
module/state-dict conversion are not part of Phase A.

Native Phase A rule records are marked `verified=True`.

## Remaining 0.1.0 surface

After the boundary and first real-Cargo route work, extend the same fail-closed
design to:

- float32 first, followed by float64 where the same semantics are verified;
- rank-1 and rank-2 tensors, plus rank-0 only if full reductions are added;
- elementwise `+`, `-`, `*`, and `/`;
- same-shape tensor operations, scalar operands, and explicitly enumerated
  trailing rank-2/rank-1 broadcasting;
- `relu`, `sigmoid`, and `tanh`;
- literal-shape `reshape` and `view`;
- literal-dimension rank-2 `transpose`;
- literal `dim` and `keepdim` forms of `sum` and `mean`;
- rank-2 `matmul`;
- `torch.nn.functional.linear` with rank-2 input and weight and optional
  rank-1 bias.

Every claim must be determined from authoritative API 1.3 type, receiver,
callable, schema, and literal metadata. Lowering must independently revalidate
the metadata and fail closed with explicit exceptions rather than assertions.

## Explicit exclusions

- CUDA, MPS, or other devices
- autograd, training, backward, optimizers, and parameter mutation
- arbitrary `nn.Module` conversion or execution
- dynamic dtype, device, rank, reduction dimensions, or reshape rank
- data-dependent Python control flow
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

## Initial acceptance checks

- Plugin entry point loads against Rextio plugin API 1.3.
- Coverage, rules, diagnostic namespace, exact crate pin and feature, type
  vocabulary, and import-minimal behavior have focused unit tests.
- Unsupported dtype/device/rank/dynamic literal/in-place cases are not claimed
  or are explicitly rejected according to the documented authority.
- Lowering has focused source/codegen tests and does not rely on `assert`.
- One Python 3.11, torch 2.11.0 real-Cargo test proves the Phase A route,
  numerical equivalence, dtype/device/shape, no-grad output, input
  non-mutation, lifetime, and fail-closed boundary rejects (**done** on the
  tested macOS arm64 environment).
- Package build plus `twine check` / `check-wheel-contents` succeeds before any
  release review (tools listed in the `dev` extra; gate not yet run for publish).

## Residual platform risks

- Certification proven on macOS arm64 only so far.
- `torch-sys` interpreter discovery depends on `PATH` / `VIRTUAL_ENV`.
- Cold native builds recompile `tch` against the active torch 2.11 install.
- Other OS/arch, CUDA/MPS, and the stretch surface remain out of scope.

## Benchmark gate

Benchmark only the generated Rextio wrapper, not a standalone tch prototype.
Compare native and eager-PyTorch fallback under `torch.no_grad`, with fixed
thread counts and warmups. Record tensor sizes, chain length, environment
versions, raw samples, and both boundary-included and internal-chain timing.
`torch.compile` is a context lane rather than the primary fallback baseline.
