# Phase B deep MLP + runtime control-flow benchmark preregistration

Date frozen: 2026-07-17  
Status: private, preregistered before Phase B timing  
Repository: `rextio/rextio-torch`

## Decision question

Phase A showed only a small advantage for a single
`linear → relu → mean(dim=1)` slice. Phase B asks whether one generated native
function becomes meaningfully faster when it keeps a deeper, weight-tied MLP
and its runtime Python loop/branch inside the native boundary.

This document freezes the code, cases, comparisons, statistics, and binary
decision rule before any Phase B performance matrix is run. Phase A source,
documents, and results remain separate and unchanged.

## Exact kernel

```python
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch.nn.functional as F


def inference(
    x: TensorF32Cpu2D,
    weight_a: TensorF32Cpu2D,
    weight_b: TensorF32Cpu2D,
    bias_a: TensorF32Cpu1D,
    bias_b: TensorF32Cpu1D,
    depth: int,
    phase: int,
) -> TensorF32Cpu1D:
    hidden = x
    for layer in range(depth):
        if (layer + phase) % 2 == 0:
            hidden = F.linear(hidden, weight_a, bias_a).relu()
        else:
            hidden = F.linear(hidden, weight_b, bias_b).relu()
    return hidden.mean(dim=1, keepdim=False)
```

`depth` and `phase` are runtime integer arguments. The two square linear
weights and biases are reused at every matching layer; they are not cloned or
unrolled. The benchmark uses only the already implemented Phase A
`F.linear`, tensor `relu`, and literal `mean(dim=1, keepdim=False)` rules.

## Frozen cells

All tensors are CPU `float32`; hidden matrices are square.

| ID | Role | Batch | Width | Depth | Phase |
|---|---|---:|---:|---:|---:|
| t1 | target | 1 | 16 | 4 | 0 |
| t2 | target | 1 | 32 | 8 | 1 |
| t3 | target | 4 | 32 | 16 | 0 |
| t4 | target | 8 | 64 | 16 | 1 |
| c1 | context | 32 | 128 | 8 | 0 |
| c2 | context | 64 | 128 | 16 | 1 |

The matrix must not be retuned after timing. Only t1–t4 decide GO/NO-GO;
c1/c2 describe scaling and can never rescue a target failure.

## Environment and fail-closed preflight

- CPython 3.11, PyTorch 2.11.0, CPU only.
- `tch =0.24.0` with `python-extension`.
- Torch intra-op and inter-op thread counts fixed at one.
- NumPy is pinned as benchmark setup data to `numpy==2.4.6`.
- `LIBTORCH_USE_PYTORCH=1`; bypassing the libtorch version check is forbidden.
- The repository and any explicitly selected local core checkout must be
  clean.
- Exact plugin and core Git SHAs, package/direct-URL provenance, harness/case
  hashes, route report hashes, build report hashes, and native artifact hash
  are recorded.
- The built route must be `native-plugin:rextio-torch`, contain exactly two
  linear claims, two ReLU claims, and one mean claim, and have no rejected
  native function.
- The imported timed native extension must be byte-identical to the artifact
  named by the build report.

Any failed required check makes the affected cell ineligible. No partial or
fallback result may be presented as Phase B evidence.

## Primary comparisons

Each cell has two independent primary paired comparisons:

1. Generated Rextio wrapper in persistent `native` mode versus the same
   generated wrapper in persistent `fallback` mode.
2. The same persistent generated `native` wrapper versus a persistent process
   importing the exact original eager kernel directly from the untouched
   project source.

Inputs, seeds, tensor objects within a request, runtime `depth`/`phase`, and
common iteration count are identical within each pair. Every lane performs
the same pre-sample collection and GC treatment. Tensor construction,
correctness materialization, process startup, native build, imports, and
first-call warm-up are outside retained timings.

For each comparison and cell:

- 9 counterbalanced paired retained samples;
- native-first and comparator-first counts differ by at most one;
- one common calibrated iteration count;
- every retained lane batch is at least 20 ms;
- GC collection precedes timing and GC is disabled symmetrically while timed;
- 10,000 deterministic paired bootstrap resamples over
  `log(native/comparator)`.

The harness retries a whole paired schedule with twice the common iteration
count if either lane falls below 20 ms. It never retains mixed iteration
counts.

## Correctness

Native, generated fallback, and exact direct eager outputs must agree by
dtype, device, rank/shape, no-grad state, and deterministic tensor digest.
Inputs must remain unmodified. The result must be CPU float32 rank 1 with
shape `(batch,)` and `requires_grad=False`.

The route/build check, artifact binding, dtype/rank/shape/no-grad checks, and
clean provenance are mandatory and fail closed.

## `torch.compile` context

`torch.compile(exact_eager_kernel, fullgraph=True)` is context-only. Per cell,
record:

- availability and correctness;
- compile plus first-call latency;
- second-call latency;
- calibrated warm loops and warm retained samples;
- warm median per-call latency.

Compilation and first/second calls are excluded from warm samples. An
unavailable compiler is reported and cannot rescue or fail either primary
comparison. If compile is available and correct for a target, the native warm
median from the native-versus-direct-eager comparison may not exceed the
compiled warm median by more than 10%.

## Frozen binary GO rule

GO requires all six cells to be route-verified, correct, and eligible for
both primary comparisons.

For **each** comparator (`fallback` and exact direct eager), all of these must
hold over t1–t4:

1. Aggregate comparator/native geometric-mean speedup is at least `1.20`.
2. Aggregate paired-bootstrap 95% confidence lower bound is at least `1.10`.
3. At least 3 of 4 individual native/comparator confidence-interval upper
   bounds are below `1.0`.
4. No target median native/comparator ratio exceeds `1.10`.

The conditional `torch.compile` rule above must also hold for every target
where compile is both available and correct. Otherwise the verdict is
NO-GO. Context cells never affect threshold calculations.

## Output ownership

Authoritative Phase B output is separate:

- `benchmarks/results_phase_b/latest.json`
- `benchmarks/results_phase_b/report.md`
- `benchmarks/results_phase_b/evidence/*`

Only a non-smoke full run from a clean committed tree may replace those
paths. Smoke checks write only to ignored temporary paths and are not
performance evidence.

## Non-claims

- No CUDA, MPS, training, autograd, optimizer, or general PyTorch claim.
- No claim beyond this exact deep alternating MLP and six frozen cells.
- No claim that `torch.compile` is a Rextio target or release requirement when
  it is unavailable.
- A Phase B GO result is evidence to consider broader private implementation;
  it is not permission to publish, tag, change repository visibility, or
  upload to PyPI.
