# Changelog

All notable changes to `rextio-torch` are documented here following Keep a
Changelog and Semantic Versioning conventions.

## [Unreleased]

- Add a Linux x86_64 **build-only**, `support_claim=false` CUDA E2 candidate
  under plugin API 1.6 and Core `rextio>=0.1.6`. New import-free
  `TensorF32Cuda0_2D` / `TensorF32Cuda0_1D` annotations carry exact
  libtorch-runtime device metadata and require the
  `rextio-device-cuda/cuda-libtorch-linux-x86_64` authorization.
- Limit CUDA lowering to rank-2 matmul, rank-2 + rank-1 bias, rank-2 ReLU, and
  literal `mean(dim=1, keepdim=False)` under no-grad. Boundary extraction and
  rank-specific return materialization verify CUDA device 0, float32, and
  rank without a CPU transfer or storage copy. They also require strided
  layout using tch 0.24's sparse/MKLDNN flags plus the exact pinned-PyTorch
  layout property at both Python boundaries.
- Add GPU-free analyzer/claim/lower/boundary tests and a Linux x86_64 Cargo
  build lane against CPython 3.11, PyTorch/libtorch 2.11.0, and tch 0.24.0.
  The lane runs actual Core profile/provider/authorization/codegen
  orchestration with the real CUDA provider and a fixed synthetic probe report,
  then compiles the generated cdylib once. It never loads or executes CUDA and
  provides no real-GPU, one-libtorch-image/ABI, numerical, performance, or
  release certification.
- Add a separate opt-in Linux x86_64/NVIDIA execution-evidence harness for the
  same frozen four-op slice. It builds the exact real provider probe, executes
  contiguous/noncontiguous `cuda:0` inputs, requires CUDA Graph replay and
  expected ATen CUDA activity, rejects transfer events, and verifies canonical
  PyTorch-wheel libtorch/c10 image identity. Its canonical offline-verifiable
  evidence remains `support_claim=false` and `certification_ready=false`; it
  is excluded from ordinary GPU-free CI and does not promote CUDA support.
- Add exact `torch.nn.functional.gelu(x)` and literal
  `approximate="none"` for float32 CPU rank-1/rank-2 inference. Both forms
  lower to fixed fallible `f_gelu("none")` under no-grad; `"tanh"`,
  dynamic/other/positional options, method/module capture, and unsupported
  types remain fail-closed. Native evidence covers default/explicit parity,
  NaN/Inf classes, signed zero, and grad-requesting input behavior.
- Add exact float32 CPU rank-1/rank-2 unary math through
  `torch.{abs,neg,negative,square,exp,log,sqrt}` and matching zero-argument
  tensor methods. Each spelling uses its exact fallible tch 0.24 API under
  no-grad, including distinct `f_neg`/`f_negative`; native certification
  compares finite values, NaN/Inf classes, signed zero, and non-mutation
  without promising portable NaN payload bits.
- Add mixed-rank matrix/vector matmul for `2D × 1D` and `1D × 2D` through
  `@`, exact `torch.matmul`, and zero-keyword `.matmul`, returning float32 CPU
  rank-1 through the existing fallible no-grad `f_matmul` helper. Rank-1 ×
  rank-1 remains fallback because rank-0 is outside the registered vocabulary.
- Add exact `torch.add/sub/mul/div(a, b)` spellings for two positional
  float32 CPU rank-1/rank-2 tensors. They reuse the existing same-rank and
  rank-2/rank-1 trailing-broadcast matrix and fallible no-grad helpers;
  scalar operands, method/alternate aliases, `alpha`, `out`, and
  `rounding_mode` remain fail-closed.
- Add exact functional activation spellings `torch.relu`, `torch.sigmoid`, and
  `torch.tanh` for the existing float32 CPU rank-1/rank-2 surface.
- Add tensor-tensor `-` and true `/` for same-rank rank-1/rank-2 tensors and
  rank-2/rank-1 trailing broadcast in either operand order. Function aliases,
  scalar operands, `rounding_mode`, and non-float32 classification results
  remain fail-closed.
- Add exact `torch.mean` / `torch.sum` function spellings and generalize method
  forms to literal dim 0/1, with keepdim omitted (False) or a named bool
  literal, only when the output remains in the existing float32 rank-1/rank-2
  vocabulary. Positional keepdim, rank-0 results, dtype, and out stay fallback.
- Add exact `torch.softmax` / `torch.argmax` function spellings and bounded
  static method variants. Argmax remains limited to exact int64 CPU rank-1
  outputs: rank-2 with keepdim=False or rank-1 dim=0 with keepdim=True. No
  int64 rank-2/rank-0 type is invented.
- Add no-bias functional linear with bias omitted, positional literal `None`,
  or keyword literal `bias=None`. Tensor-valued keyword operands remain
  ordinary fallback because Core plugin API 1.3 cannot represent them.
- Independently revalidate canonical target, arity, operand/literal alignment,
  keyword name/type/value, result type, and receiver shape at lowering. Static
  positional dim/None metadata is never forwarded as a runtime tensor operand.
- Extend the serialized native Cargo project with a complete CPU follow-up
  slice and all subtraction/division rank-order families, proving route,
  numerical parity, dtype/device/rank, no-grad, non-mutation, and incompatible
  broadcast failure under the unchanged PyTorch 2.11.0 / tch 0.24.0 pins.
- Add binary elementwise `a * b` for float32 CPU rank-1/rank-2 tensors: same
  rank and rank-2/rank-1 trailing broadcast in either order. It is binop-only;
  scalar operands, `torch.mul`, `.mul`, int64 classification results, and all
  other dtypes/devices/ranks remain fail-closed.
- Lower multiplication through a fallible `tch::Tensor::f_mul` helper under
  `no_grad`, with claim/lower metadata revalidation and real-Cargo coverage.

## [0.1.2] - Unreleased

- Add the inference-only rank-2 classification-head method chain:
  `.softmax(dim=1).argmax(dim=1, keepdim=False)`. The emitted tch 0.24.0
  helpers run under `no_grad`, and the final result is checked/materialized as
  an int64 CPU rank-1 tensor.
- Add `TensorI64Cpu1D` exclusively for that classification result boundary.
  Functional spellings, dtype overrides, dynamic dimensions/keepdim, rank-3+,
  non-CPU tensors, training/autograd, and in-place forms remain fail-closed.
- Keep the pre-existing add rules explicitly float32-only now that the type
  vocabulary includes an int64 result: int64+int64 and mixed int64/float32
  operands are rejected at claim time and independently at lowering.
- Extend native Cargo certification with numerical, dtype, shape, no-grad,
  non-mutation, and route evidence for the classification head.
- Update the normal CI triggers to include the active `0.1.2` integration
  branch. The package remains unreleased: no tag or publication is implied.

## [0.1.1] - Unreleased integration baseline

- Compatibility hotfix for Core 0.1.5 / plugin API 1.4 hosts: provider methods
  no longer require the host's `PLUGIN_API_VERSION` to equal the provider's
  declared API 1.3; Core's loader owns compatibility negotiation.
- Fail closed at provider registration unless the host API is parseable
  as major 1 with minor >=3, preventing bypassed dependencies from admitting
  Core API 1.2 or incompatible/malformed versions.
- Retain the declared provider API at 1.3 and the existing `rextio`, PyTorch,
  and `tch` pins. No standalone-artifact capability is declared.
- Reject standalone/non-PyO3 lowering contexts while retaining compatibility
  with legacy contexts that omit `backend`.

## [0.1.0] - 2026-07-18 (public Alpha)

This is the first public native-AOT Alpha release of `rextio-torch`.

The public-source **Alpha AOT** release expands the Phase A vertical slice into a
broader fail-closed native surface. **Certified** real-Cargo evidence remains
**macOS arm64 / CPython 3.11 / torch 2.11.0** (`LIBTORCH_USE_PYTORCH=1`; no
version-check bypass). **Linux x86_64** and **Linux AArch64** are
**experimental** runtime-backed hosts; **macOS x86_64** is availability-gated
because the pinned torch wheel does not exist; 32-bit Linux/macOS cells are
explicitly unsupported; **Windows** is **deferred** (unverified).

Phase A/B benchmark result artifacts are retained as **historical context
only**; performance is not a release gate for this Alpha cut.

### Added

- Public Alpha package metadata for plugin API 1.3; the temporary upload-block
  classifier was removed only after the release passed its blocking
  CI matrix.
- `requires-python = ">=3.11,<3.12"` (CPython 3.11 only for this cut).
- Materialized `RxtTorchTensor` boundary types for float32 CPU rank-1 and
  rank-2 tensors, with PyO3 0.29-compatible raw tch `python-extension` bridge
  helpers (unpack/wrap without copying storage; clone via `shallow_clone`).
- Phase A vertical slice (certified): `torch.nn.functional.linear` on float32
  CPU rank-2 input/weight plus rank-1 bias, method `.relu()`, and method
  `.mean(dim=1, keepdim=False)` reducing to float32 CPU rank-1.
- **Alpha AOT surface** on the same fail-closed claim/lower/helper architecture:
  - `.relu()` / `.sigmoid()` / `.tanh()` for float32 CPU rank-1 and rank-2
  - rank-2 matmul via `@`, `torch.matmul`, and method `.matmul`
  - elementwise `+` for same-rank tensors and rank-2 + rank-1 trailing bias
    broadcast (either order); concrete sizes remain runtime-checked by tch
  - `.sum(dim=1, keepdim=False)` with the same literal guards as mean
- Fail-closed claim and independent lower revalidation from authoritative API
  1.3 type, receiver, keyword, and literal metadata.
- Exact crate pin `tch =0.24.0` with the `python-extension` feature; public
  core floor `rextio>=0.1.3,<0.2`; pinned `torch==2.11.0`.
- Focused unit tests for coverage, rules, types, import-minimal discovery,
  claim, lower, analyzer integration (Phase A chain, Phase B deep control,
  Alpha AOT control-flow surface).
- Real-Cargo E2E certification for the Phase A chain and an Alpha AOT
  control-flow slice (scalar Python `for`/`if` → Rust control flow around tch calls):
  native route evidence, numerical equivalence to eager PyTorch, CPU/float32/rank,
  no-grad output with grad-requesting inputs, input non-mutation, and
  fail-closed boundary rejects (float64 / wrong rank) with stable
  `rextio-torch:` messages.
- The same serialized Alpha Cargo project compiles and executes binary,
  functional, and method matmul; same-rank and rank-2/rank-1 add; sum/mean;
  rank-1/rank-2 relu, sigmoid, and tanh. Tensor sizes remain a runtime libtorch
  contract because Alpha annotations encode rank rather than concrete shapes.
- Native Phase A and expanded Alpha AOT rule records marked `verified=True`;
  every advertised rule family is compiled and executed by the serialized
  certification project and carries a unique diagnostic code.
- Exact `check-wheel-contents==0.6.3` in the `dev` extra for the package gate.
- Explicit **certified / experimental / deferred** host-platform semantics in
  the README and implementation plan (macOS arm64 certified; Linux x86_64 and
  aarch64 experimental; Windows deferred), plus a Linux experimental smoke
  recipe that keeps `LIBTORCH_USE_PYTORCH=1` and forbids
  `LIBTORCH_BYPASS_VERSION_CHECK`.
- Opt-in maintainer script `scripts/linux-smoke.sh` and focused unit tests for
  the portable pin/host contract (plugin must not reject Linux solely by OS;
  real-Cargo e2e remains optional and fail-closed on pin mismatches).
- A declarative eight-cell Linux/macOS architecture truth model (`x86` = i686,
  `x64` = x86_64, ARM32 = ARMv7, ARM64 = AArch64) with stable fail-closed
  reasons for unavailable pinned runtimes.
- SHA-pinned, read-only GitHub Actions for quality, all platform-contract
  cells, real native E2E on macOS ARM64 and Linux x86_64, package artifacts and
  clean install smoke; scheduled/manual evidence covers Linux AArch64 and the
  macOS x86_64 artifact-availability gate.
- A stable `CI gate` branch-protection result, exact package-build tool pins,
  non-isolated artifact builds, and wheel-installed (not editable-installed)
  native E2E with an explicit zero-skip assertion for runtime-backed profiles.
- Placeholder sanitization for workstation paths in retained Phase A/B raw
  benchmark provenance; measurements and evidence lineage are unchanged.
- An explicit source-distribution manifest containing the shipped test suite,
  benchmark fixtures, documentation, and Linux smoke script, plus a CI audit
  that proves repository-byte equality and runs focused tests after extraction.

### Notes

- Literal transpose / reshape / view remain **unclaimed** (view/alias risk with
  shallow-clone boundary ownership).
- Core does not offer method claims whose receiver is a bare BinOp expression;
  write named temps (e.g. `even = a @ b + bias; hidden = even.relu()`), with
  distinct names per if-arm when both branches need intermediates.
