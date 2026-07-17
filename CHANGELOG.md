# Changelog

All notable changes to `rextio-torch` are documented here following Keep a
Changelog and Semantic Versioning conventions.

## [Unreleased]

Private incubator cut. **Alpha AOT** expands the Phase A vertical slice into a
broader fail-closed native surface on the tested **macOS arm64 / CPython 3.11 /
torch 2.11.0** environment (`LIBTORCH_USE_PYTORCH=1`; no version-check bypass).
Package remains unreleased (`Private :: Do Not Upload`).

Phase A/B benchmark result artifacts are retained as **historical context
only**; performance is not a release gate for this Alpha cut.

### Added

- Private incubator package scaffold for plugin API 1.3 with the
  `Private :: Do Not Upload` classifier.
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
- `check-wheel-contents>=0.6` in the `dev` extra for the package gate.

### Notes

- Literal transpose / reshape / view remain **unclaimed** (view/alias risk with
  shallow-clone boundary ownership).
- Core does not offer method claims whose receiver is a bare BinOp expression;
  write named temps (e.g. `even = a @ b + bias; hidden = even.relu()`), with
  distinct names per if-arm when both branches need intermediates.
