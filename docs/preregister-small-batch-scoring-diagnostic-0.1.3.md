# Diagnostic small-batch scoring pipeline (0.1.3 candidate)

**Status:** diagnostic only — **not** an official performance cohort

**Does not** overwrite Phase A/B historical results under
`benchmarks/results/` or `benchmarks/results_phase_b/`

**Makes no speedup claim**

## Purpose

Exercise a product-shaped multi-op inference pipeline at small batches while
keeping correctness gates strict enough that argmax cannot hide drift:

1. Rank-2 / rank-1 normalization by subtraction then division
2. Small functional linear
3. Scalar `for layer in range(rounds)` with integer `if` routing through
   ReLU vs tanh (`rounds=4` control-flow iterations on every predeclared cell)
4. Classification linear
5. Softmax
6. Argmax

## Predeclared matrix

| Case | batch | features | hidden | classes | rounds (control-flow) | phase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s1 | 1 | 32 | 32 | 8 | 4 | 0 |
| s16 | 16 | 32 | 32 | 8 | 4 | 0 |
| s128 | 128 | 32 | 32 | 8 | 4 | 0 |

Source of truth: `benchmarks/small_batch_scoring_cases.py`.

`rounds` is the kernel/control-flow iteration count. Diagnostic **timing
sample** counts are separate (`DEFAULT_TIMING_SAMPLES` / `--timing-samples`)
and must be a positive integer.

## Correctness protocol

For each cell, compare native (when a built project is supplied) against an
eager reference **in this order**:

1. **Full logits** — `torch.allclose` with `rtol=1e-5`, `atol=1e-6`
2. **Full probabilities** — same numeric tolerance
3. **Exact labels** — integer equality only after (1) and (2) pass

Argmax-only comparison is forbidden as a sole gate.

## Diagnostic lanes

| Lane | Role | When measured |
| --- | --- | --- |
| `native_rextio` | primary diagnostic native | Only when an **already-built** project is supplied (`--built-project` / `load_native_scoring_lane`). Wrappers are imported and retained **once** outside timed samples; route/provenance is verified; no rebuild inside timing. |
| `eager` | context | Always when torch is available |
| `torch.inference_mode` | context | When `torch.inference_mode` exists |

Notes:

- PyTorch’s **default grad mode is enabled**. These diagnostic inputs use
  `requires_grad=False` tensors; that is not the same as “grad mode off.”
- The ordinary CLI **does not** build native code and must mark
  `native_rextio` unavailable with an explicit reason when no built project
  is supplied. It must **never** silently imply native was measured.
- All timings are labeled **`diagnostic`**. This harness has **no** expansion
  GO gate and records **`speedup_claim: null`**.

## Invocation-scope / no-grad note

Production native helpers still use **per-operation** `tch::no_grad_guard()`.
Core API 1.6 has no function-body prelude hook; see
[invocation-scope-proposal-0.1.3.md](invocation-scope-proposal-0.1.3.md).
This diagnostic must not claim a single RAII scope per invocation is active.

## Outputs

| Mode | Writes |
| --- | --- |
| default / smoke | temp or ignored path only |
| official cohort | **not supported** — refuse to write under historical result trees |

## Commands

```bash
# Harness contracts only (no builds, no timing)
.venv/bin/python -m pytest tests/test_small_batch_scoring_harness.py \
  tests/test_analyzer_small_batch_scoring.py -q

# Context-only diagnostic run (native explicitly unavailable)
.venv/bin/python -m benchmarks.bench_small_batch_scoring --smoke

# Native diagnostic against an already-built project root
.venv/bin/python -m benchmarks.bench_small_batch_scoring \
  --built-project /path/to/built/project --smoke
```

## Explicit non-claims

- No official performance cohort and no recorded speedup gate.
- No CUDA / MPS / training.
- No claim that invocation-scope no_grad optimization is active.
- Historical Phase A/B JSON and reports remain untouched.
- Absence of `--built-project` means native was **not** measured.
