# Phase A product-route benchmark

Preregistered harness for the float32 CPU `linear → ReLU → mean(dim=1)` product
route. Primary comparison calls the **same generated Rextio wrapper** in two
**persistent processes**; only `REXTIO_NATIVE_MODE` differs (`native` vs
`fallback`). Direct eager PyTorch and `torch.compile(fullgraph=True)` are
**context-only** lanes and never enter the expansion GO gate.

Protocol (frozen before measurement):
[`docs/preregister-phase-a-benchmark-2026-07-17.md`](../docs/preregister-phase-a-benchmark-2026-07-17.md).

## Do not run yet

This repository ships the harness and unit tests only. **No performance
benchmark has been executed here**, and no authoritative result files exist
until a deliberate full run is performed under the preregistered pins.

## Commands (when ready)

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

## Expansion GO gate (binary)

All of the following must hold or the verdict is **NO-GO** for broadening /
public release now. Context sizes and context lanes **cannot** rescue a
failure:

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

After the preregistration and harness are committed, a deliberate clean-tree
run may use:

```bash
# Fast preflight only; never authoritative evidence
.venv/bin/python -m benchmarks.bench_phase_b --smoke

# Full preregistered matrix; owns only results_phase_b/
.venv/bin/python -m benchmarks.bench_phase_b
```
