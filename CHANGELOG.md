# Changelog

All notable changes to `rextio-torch` are documented here following Keep a
Changelog and Semantic Versioning conventions.

## [Unreleased]

No changes yet.

## [0.1.0] - 2026-07-18 (public Alpha release candidate)

This dated section records the reviewed 0.1.0 candidate; it does **not** claim
that PyPI publication has already occurred. Publication, tag, and live-install
verification remain release-owner actions after this candidate is approved.

The public-source **Alpha AOT** candidate expands the Phase A vertical slice into a
broader fail-closed native surface. **Certified** real-Cargo evidence remains
**macOS arm64 / CPython 3.11 / torch 2.11.0** (`LIBTORCH_USE_PYTORCH=1`; no
version-check bypass). **Linux x86_64** and **Linux AArch64** are
**experimental** runtime-backed hosts; **macOS x86_64** is availability-gated
because the pinned torch wheel does not exist; 32-bit Linux/macOS cells are
explicitly unsupported; **Windows** is **deferred** (unverified). The package
is a release candidate whose live PyPI publication is still pending.

Phase A/B benchmark result artifacts are retained as **historical context
only**; performance is not a release gate for this Alpha cut.

### Added

- Public Alpha package metadata for plugin API 1.3; the temporary upload-block
  classifier was removed only after the merged candidate passed its blocking
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
