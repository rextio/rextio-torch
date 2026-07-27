# Phase A product-route benchmark

Preregistered harness for the float32 CPU `linear → ReLU → mean(dim=1)` product
route. Primary comparison calls the **same generated Rextio wrapper** in two
**persistent processes**; only `REXTIO_NATIVE_MODE` differs (`native` vs
`fallback`). Direct eager PyTorch and `torch.compile(fullgraph=True)` are
**context-only** lanes and never enter the expansion GO gate.

Protocol (frozen before measurement):
[`docs/preregister-phase-a-benchmark-2026-07-17.md`](../docs/preregister-phase-a-benchmark-2026-07-17.md).

## Historical execution status (supersedes the preregistration-era warning)

The earlier “do not run yet / no benchmark executed” notice applied only
before the frozen runs on 2026-07-17. Phase A and Phase B were subsequently
executed, and their authoritative JSON, reports, and route/build evidence are
retained in this repository.

- Phase A verdict: **NO-GO** for performance-gated expansion (aggregate
  fallback/native speedup `1.098517`, below the preregistered `1.20` gate).
- Phase B verdict: **NO-GO** against both fallback (`1.151046`) and direct eager
  (`1.134275`), each below the same aggregate speedup gate.

These measurements remain historical evidence and performance-NO-GO context.
They do not block the later, narrowly scoped native-AOT Alpha, whose objective
superseded speed-gated expansion. No further benchmark run is requested for
the 0.1.0 release train.

## Reproduction commands (historical; not routine release checks)

The commands are retained for reproducibility. Do not overwrite authoritative
results without a separately approved, preregistered rerun.

```bash
# Fast harness / route gate only — writes to ignored temp paths only
.venv/bin/python -m benchmarks.bench_phase_a --smoke

# Full preregistered matrix — atomically writes tracked results
.venv/bin/python -m benchmarks.bench_phase_a
```

## Frozen matrix

| Case | Role | batch | in | out |
| --- | --- | ---: | ---: | ---: |
| t1 | target | 1 | 32 | 32 |
| t2 | target | 1 | 128 | 64 |
| t3 | target | 8 | 64 | 64 |
| t4 | target | 16 | 128 | 64 |
| c1 | context size | 64 | 128 | 64 |
| c2 | context size | 256 | 256 | 128 |

All float32 CPU with fixed deterministic seeds for `x` / `weight` / `bias`.

## Method (summary)

- Build once; verify `check.json` route `native-plugin:rextio-torch` and
  `build.json` native status `built`; hash reports and the native artifact.
- Import **torch before** the native extension; CPython **3.11**, torch
  **2.11.0**, plugin API **1.3**, `tch =0.24.0` with `python-extension`,
  `LIBTORCH_USE_PYTORCH=1`, matching `VIRTUAL_ENV` / `PATH` / `PYO3_PYTHON`,
  **never** `LIBTORCH_BYPASS_VERSION_CHECK`.
- Torch intra-op and inter-op threads fixed to **1**; fixed harness seed.
- Common iteration calibration so **both** lanes retain samples ≥ **20 ms**;
  GC collect before samples, GC disabled during timing.
- **9** deterministic counterbalanced paired samples per cell; raw schedules
  and samples retained.
- **10,000**-resample paired bootstrap over `log(native/fallback)`.
- Compilation and first-call warm-up reported outside steady-state.

## Historical expansion GO gate (binary; superseded as a release criterion)

Under the original Phase A decision protocol, all of the following had to hold
or the verdict was **NO-GO** for performance-gated broadening/publication.
Context sizes and context lanes could not rescue a failure:

1. Every primary cell (all six) is provenance/correctness eligible.
2. Target aggregate geometric-mean **fallback/native** speedup point estimate
   ≥ **1.20**, and paired-bootstrap 95% lower bound ≥ **1.10**.
3. At least **3 of 4** target cells have the 95% CI **upper** bound for
   **native/fallback** below **1.0**.
4. No target cell median **native/fallback** ratio exceeds **1.10**.

## Outputs

| Mode | Writes |
| --- | --- |
| full | atomic `benchmarks/results/latest.json`, `report.md`, `evidence/{check,build}.json` |
| smoke | ignored/temp path only — never overwrites authoritative results |

## Explicit non-claims

- This Mac / recorded machine only.
- Boundary-inclusive product latency (no internal-only native timing).
- No CUDA / MPS / training.
- No extrapolation beyond the six frozen cells and recorded pins.
- Smoke is not performance evidence.

## Unit tests

```bash
.venv/bin/python -m pytest tests/test_benchmark_harness.py -q
```

These tests use fakes only; they do **not** run smoke or full timing.

## Phase B: deep MLP with mixed Python control flow

Phase B is an independent, preregistered follow-up. Its exact kernel keeps a
runtime `for` loop and alternating `if` branch around two weight-tied
`linear → ReLU` paths, followed by `mean(dim=1, keepdim=False)`. The frozen
six-cell matrix and both primary comparisons are specified in
[`docs/preregister-phase-b-deep-control-benchmark-2026-07-17.md`](../docs/preregister-phase-b-deep-control-benchmark-2026-07-17.md).

Phase B does not overwrite or reinterpret Phase A evidence:

- authoritative JSON: `benchmarks/results_phase_b/latest.json`
- authoritative report: `benchmarks/results_phase_b/report.md`
- copied route/build evidence: `benchmarks/results_phase_b/evidence/`

The focused unit tests exercise frozen cases, output ownership, and synthetic
GO/NO-GO decisions only. They do not build an extension, start benchmark
workers, or collect timings:

```bash
.venv/bin/python -m pytest tests/test_analyzer_phase_b_chain.py \
  tests/test_phase_b_benchmark_harness.py -q
```

The following Phase B commands are likewise retained for reproduction only;
they are not an instruction to rerun benchmarks during the Alpha release:

```bash
# Fast preflight only; never authoritative evidence
.venv/bin/python -m benchmarks.bench_phase_b --smoke

# Full preregistered matrix; owns only results_phase_b/
.venv/bin/python -m benchmarks.bench_phase_b
```

## Diagnostic small-batch scoring (0.1.3)

Self-contained product-shaped pipeline for local diagnosis only. **Not** an
official performance cohort. **Does not** write under `results/` or
`results_phase_b/`. **Makes no speedup claim.**

Protocol:
[`docs/preregister-small-batch-scoring-diagnostic-0.1.3.md`](../docs/preregister-small-batch-scoring-diagnostic-0.1.3.md).

Predeclared cells: batch **1 / 16 / 128**, features **32**, classes **8**,
control-flow **`rounds=4`** (`for layer in range(rounds)`). Timing sample
count is separate (`--timing-samples`, default 4) and must be positive.
Correctness order: full logits → full probabilities (numeric tolerance) →
exact labels.

Lanes:

- **`native_rextio`** (primary diagnostic) only when an already-built project
  is supplied; wrappers are retained once; no rebuild inside timed samples;
  ordinary CLI marks native unavailable unless `--built-project` is set.
- **eager** and **`torch.inference_mode`** are context lanes.

PyTorch default grad mode remains enabled; diagnostic inputs use
`requires_grad=False`. Eligible API 1.7 native functions use one
function-scope `no_grad_guard`; RXT075/legacy/type-only paths retain local
per-operation guards
(`docs/invocation-scope-proposal-0.1.3.md`).

```bash
# Contracts only
.venv/bin/python -m pytest tests/test_small_batch_scoring_harness.py \
  tests/test_analyzer_small_batch_scoring.py -q

# Context-only diagnostic (native explicitly unavailable)
.venv/bin/python -m benchmarks.bench_small_batch_scoring --smoke

# Native diagnostic against a prebuilt project root
.venv/bin/python -m benchmarks.bench_small_batch_scoring \
  --built-project /path/to/built/project --smoke
```
