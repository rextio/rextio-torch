# Preregistration: Phase A product-route expansion benchmark

Status: **PREREGISTERED BEFORE MEASUREMENT — 2026-07-17**. This document freezes
the benchmark matrix, environment pins, statistical protocol, and binary
expansion GO gate **before** any performance samples are collected. No result
section is present yet; filling results requires a later deliberate full run
under this exact protocol.

Harness implementation: `benchmarks/cases.py`, `benchmarks/bench_phase_a.py`,
`benchmarks/README.md`.

## Decision being tested

Phase A (float32 CPU `linear → ReLU → mean(dim=1, keepdim=False)`) is
implemented and real-Cargo certified. The expansion / public-release question
is whether the **generated Rextio product wrapper** is fast enough on a frozen
target niche, under honest product-route latency, to justify broadening or
shipping a performance claim.

Failure of the GO gate means **NO-GO for broadening / public release now**.
Context sizes and context lanes cannot rescue a failure.

## Frozen product surface

```python
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch.nn.functional as F

def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
```

Primary comparison: the **same** generated Rextio wrapper in two **persistent**
processes. Only `REXTIO_NATIVE_MODE` changes (`native` vs `fallback`). Build
once. Compilation and first-call warm-up stay outside steady-state samples.

## Frozen environment pins

| Pin | Value |
| --- | --- |
| CPython | 3.11 only |
| torch | 2.11.0 |
| Plugin API | 1.3 |
| rextio | `>=0.1.3,<0.2` |
| tch | `=0.24.0` with feature `python-extension` |
| LibTorch link | `LIBTORCH_USE_PYTORCH=1` |
| Version bypass | **forbidden** (`LIBTORCH_BYPASS_VERSION_CHECK` must not be set) |
| Interpreter wiring | matching `VIRTUAL_ENV`, `PATH`, `PYO3_PYTHON` |
| Torch threads | `intra-op=1`, `inter-op=1` |
| Device / dtype | CPU float32 only |

Import **torch before** loading the generated native extension. Provenance must
be clean and exact (plugin + core install mode / direct URL / git SHA when
applicable). Route evidence: verified `check.json` / `build.json` with
`native-plugin:rextio-torch`, accepted native status, native artifact hash,
harness + cases SHA-256.

## Frozen case matrix (exactly six)

| Case | Role | batch | in_features | out_features |
| --- | --- | ---: | ---: | ---: |
| t1 | target niche | 1 | 32 | 32 |
| t2 | target niche | 1 | 128 | 64 |
| t3 | target niche | 8 | 64 | 64 |
| t4 | target niche | 16 | 128 | 64 |
| c1 | context size | 64 | 128 | 64 |
| c2 | context size | 256 | 256 | 128 |

Fixed deterministic seeds derive from base `20260717` plus the case index for
`x`, `weight`, and `bias`. No cell may be added, removed, or retuned after
observing timings.

## Correctness gates (every primary cell)

Before a cell contributes eligible evidence:

- native and fallback correctness digests match;
- dtype float32, device CPU, shape `(batch,)`, `requires_grad is False`;
- inputs are not mutated;
- route / build / provenance verified.

Any failure records blocking reasons and keeps the cell ineligible.

## Timing protocol

- Persistent workers: one native process, one fallback process.
- Common iteration calibration so **both** lanes retain samples ≥ **20 ms**.
- If any retained sample falls under 20 ms, double the common iteration count
  and recollect **all** pairs for that cell (up to four retries).
- GC: collect before each timed batch; disable GC symmetrically during timing.
- **9** deterministic counterbalanced paired samples per cell (native-first /
  fallback-first differ by at most one; extra first lane seed-assigned; raw
  schedule retained).
- Paired bootstrap: **10,000** resamples over `log(native/fallback)` per cell;
  aggregate bootstrap resamples within each target cell then geomeans the
  four-cell speedup.
- Raw schedules, samples, medians, bootstrap blocks, and warm-up times retained.

## Context-only lanes (not GO-gate inputs)

For each measured cell, also record:

1. **Direct eager** PyTorch of the same graph.
2. **`torch.compile(..., fullgraph=True)`**, recording compile / first-call time,
   a second-call edge, and warm steady-state samples.

If a context lane is unavailable, report that honestly. Context lanes **must
never** affect the Rextio expansion GO gate.

## Binary expansion GO gate

**GO** only if **all** hold; otherwise **NO-GO**:

1. **Every** primary cell (all six) is provenance/correctness eligible.
2. Target aggregate geometric-mean **fallback/native** speedup point estimate
   ≥ **1.20**, and the paired-bootstrap 95% **lower** bound on that aggregate
   speedup ≥ **1.10**.
3. At least **3 of 4** target cells individually have the 95% CI **upper** bound
   for **native/fallback** **below 1.0**.
4. **No** target cell median **native/fallback** ratio exceeds **1.10**.

Any failure is **NO-GO for broadening / public release now**. Context sizes
(c1, c2) and context lanes cannot rescue it.

## Outputs

| Mode | Ownership |
| --- | --- |
| full | Atomic tracked `benchmarks/results/latest.json`, human `report.md`, retained `evidence/check.json` + `evidence/build.json` |
| smoke | Ignored/temp paths only; must not overwrite authoritative results; not performance evidence |

## Explicit non-claims

- **This Mac only** (recorded machine / platform); no multi-host generalization.
- **Boundary-inclusive product latency** through the generated wrapper — not
  internal-only native kernel timing.
- **No CUDA / MPS / training** claims.
- **No extrapolation** beyond the six frozen cells, fixed pins, and recorded
  seeds.
- Smoke / incomplete runs are harness checks only.

## Unit tests (no performance run)

`tests/test_benchmark_harness.py` covers case freeze, scheduling, calibration
retry logic, bootstrap determinism, gate computation, smoke/full output
ownership, and fail-closed validation with fakes. Those tests must not execute
smoke or full timing.
