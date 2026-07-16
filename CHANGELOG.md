# Changelog

All notable changes to `rextio-torch` are documented here following Keep a
Changelog and Semantic Versioning conventions.

## [Unreleased]

Private incubator cut. Phase A is **implemented and real-Cargo certified** on
the tested **macOS arm64 / CPython 3.11 / torch 2.11.0** environment
(`LIBTORCH_USE_PYTORCH=1`; no version-check bypass). Package remains unreleased
(`Private :: Do Not Upload`).

### Added

- Private incubator package scaffold for plugin API 1.3 with the
  `Private :: Do Not Upload` classifier.
- `requires-python = ">=3.11,<3.12"` (CPython 3.11 only for this cut).
- Materialized `RxtTorchTensor` boundary types for float32 CPU rank-1 and
  rank-2 tensors, with PyO3 0.29-compatible raw tch `python-extension` bridge
  helpers (unpack/wrap without copying storage; clone via `shallow_clone`).
- Phase A vertical slice: `torch.nn.functional.linear` on float32 CPU rank-2
  input/weight plus rank-1 bias, method `.relu()`, and method
  `.mean(dim=1, keepdim=False)` reducing to float32 CPU rank-1.
- Fail-closed claim and independent lower revalidation from authoritative API
  1.3 type, receiver, keyword, and literal metadata.
- Exact crate pin `tch =0.24.0` with the `python-extension` feature; public
  core floor `rextio>=0.1.3,<0.2`; pinned `torch==2.11.0`.
- Focused unit tests for coverage, rules, types, import-minimal discovery,
  claim, lower, and analyzer integration for the nested claim chain.
- One real-Cargo E2E certification for the Phase A chain: native route
  evidence, numerical equivalence to eager PyTorch, CPU/float32/rank-1 output,
  no-grad output with grad-requesting inputs, input non-mutation, returned
  tensor lifetime, and fail-closed boundary rejects (float64 / wrong rank)
  with stable `rextio-torch:` messages.
- Native Phase A rule records marked `verified=True`.
- `check-wheel-contents>=0.6` in the `dev` extra for the package gate.
