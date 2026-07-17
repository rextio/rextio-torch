# rextio-torch

**Private incubator** Rextio plugin that lowers a proven subset of Python
PyTorch inference code to Rust expressions backed by
[`tch`](https://github.com/LaurentMazare/tch-rs).

Status: **unreleased 0.1.0 Alpha** — not for PyPI. Package metadata includes
`Private :: Do Not Upload`.

**Alpha AOT** expands the original Phase A vertical slice into a broader
fail-closed, real-Cargo-certified native surface on the tested environment: **macOS arm64**,
**CPython 3.11**, **torch 2.11.0**, Rextio plugin API **1.3**, with
`LIBTORCH_USE_PYTORCH=1` (never `LIBTORCH_BYPASS_VERSION_CHECK`).

Performance benchmarks from Phase A/B are retained only as **historical
context** under `benchmarks/results/` and `benchmarks/results_phase_b/`. They
are **not** a release gate for this Alpha cut.

## Compatibility baseline

| Component | Pin |
| --- | --- |
| CPython | 3.11 only (`requires-python = ">=3.11,<3.12"`) |
| Rextio | `>=0.1.3,<0.2` (plugin API **1.3**) |
| PyTorch | `torch==2.11.0` |
| Rust binding | `tch==0.24.0` with feature `python-extension` |
| Device | **CPU only** |
| Mode | **Inference / no-grad only** |

The published `tch` 0.24.0 release targets libtorch / PyTorch **2.11.0**
exactly. `LIBTORCH_BYPASS_VERSION_CHECK` is **not** an accepted build path.
Native builds that link the Python PyTorch install set `LIBTORCH_USE_PYTORCH=1`
and must use the same active CPython 3.11 environment as Rextio / PyO3.

## Alpha AOT surface

Float32 CPU rank-1 and rank-2 tensors, claimed only when API 1.3 metadata
proves the form. Every claimed site independently revalidates lowering
metadata, uses fallible `tch` APIs under `no_grad`, avoids `assert`, and keeps
zero-storage-copy Python tensor boundaries.

| Form | Notes |
| --- | --- |
| `torch.nn.functional.linear(x, w, b)` | Rank-2 input/weight + rank-1 bias (Phase A certified) |
| `.relu()` / `.sigmoid()` / `.tanh()` | Method form, rank-1 and rank-2 |
| `a @ b` / `torch.matmul` / `.matmul` | Rank-2 × rank-2 → rank-2; inner sizes checked by tch at runtime |
| `a + b` | Same-rank 1d/2d or rank-2 + rank-1; concrete broadcast sizes checked by tch at runtime |
| `.mean(dim=1, keepdim=False)` | Rank-2 → rank-1 (Phase A certified) |
| `.sum(dim=1, keepdim=False)` | Rank-2 → rank-1 (same literal guards as mean) |

### Example (Phase A certified chain)

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

### Example (control-flow vertical slice)

Core claims method calls on named receivers and call chains; a bare BinOp
receiver such as `(a @ b + bias).relu()` is not offered to plugins. Use temps:

```python
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D

def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
    depth: int,
    phase: int,
) -> TensorF32Cpu1D:
    hidden = x
    for layer in range(depth):
        if (layer + phase) % 2 == 0:
            even = hidden @ weight + bias
            hidden = even.relu()
        else:
            odd = hidden @ weight + bias
            hidden = odd.sigmoid()
    return hidden.mean(dim=1, keepdim=False)
```

Python `for` / `if` with Rextio-lowerable scalar `int` / `bool` conditions become
Rust control flow around native tch helpers. Tensor comparisons and
tensor-data-dependent conditions are not claimable by plugin API 1.3 and remain
on the Python fallback.

Tensor annotations prove dtype, device, and rank only. They do not carry
concrete dimensions, so matmul compatibility and concrete PyTorch broadcasting
are deliberately validated by the reused libtorch runtime at execution time.
One serialized certification project exercises every Alpha rule family,
including both call forms of matmul, both supported add rank patterns, sum,
tanh, and rank-1 activation chaining.

### Annotation vocabulary (`rextio_torch.types`)

| Annotation | dtype | device | rank |
| --- | --- | --- | --- |
| `TensorF32Cpu2D` | float32 | CPU | 2 |
| `TensorF32Cpu1D` | float32 | CPU | 1 |

Marker classes intentionally **import neither torch nor Rextio**. Runtime
values remain ordinary `torch.Tensor` objects; the analyzer resolves the dotted
spellings when the plugin is enabled.

### Explicit exclusions (still fallback / rejected)

CUDA/MPS, training/autograd, arbitrary `nn.Module`, dynamic dtype/device/rank
or reduction dimensions, in-place ops, scalar/`*`/`-`/`/` elementwise forms,
literal transpose/views (not claimed — view/alias risk), unsupported
broadcasting rank combinations, tensor comparisons/data-dependent conditions,
and version-check bypasses.

Unsupported forms must remain fallback or rejected — never falsely claimed.

## Install (development only)

```bash
# CPython 3.11 only; requires rextio 0.1.3+ and torch 2.11.0
python3.11 -m pip install -e '.[dev]'
```

Plugin discovery and `rextio_torch.types` import **without** importing torch.
Real native builds need a matching torch 2.11 / libtorch environment and a Rust
toolchain. The `dev` extra includes packaging tools (`build`, `twine`,
`check-wheel-contents`) for the pre-release package gate.

## Historical benchmarks (not a release gate)

Phase A and Phase B product-route benchmarks were preregistered and executed
with fixed cells. Recorded numbers live under:

- `benchmarks/results/` (Phase A)
- `benchmarks/results_phase_b/` (Phase B deep control)

Those runs measured boundary-inclusive latency and expansion GO criteria. For
this Alpha cut they are **historical evidence only** — usefulness of the pinned
AOT surface is the product goal, not beating a speedup threshold.

## Residual platform risks

- `torch-sys` discovers Python via `PATH` / `VIRTUAL_ENV`, not only
  `PYO3_PYTHON`; a mismatched torch on `PATH` fails the exact 2.11.0 check.
- Cold builds compile `tch` / `torch-sys` for each generated project.
- The native extension must load under the same CPython 3.11 + torch 2.11.0
  that built it.
- Certification was executed on **macOS arm64**; other OS/arch combinations are
  not yet proven.
- Package remains private / unreleased; do not publish without a release review.

## Repository safeguards

- Do not publish to PyPI or change GitHub visibility during incubation.
- Do not use `LIBTORCH_BYPASS_VERSION_CHECK`.
- Do not add a project-local `AGENTS.md` without owner direction.
