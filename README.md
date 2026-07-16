# rextio-torch

**Private incubator** Rextio plugin that lowers a proven subset of Python
PyTorch inference code to Rust expressions backed by
[`tch`](https://github.com/LaurentMazare/tch-rs).

Status: **unreleased 0.1.0** — not for PyPI. Package metadata includes
`Private :: Do Not Upload`.

**Phase A is implemented and real-Cargo certified** on the tested environment:
**macOS arm64**, **CPython 3.11**, **torch 2.11.0**, Rextio plugin API **1.3**,
with `LIBTORCH_USE_PYTORCH=1` (never `LIBTORCH_BYPASS_VERSION_CHECK`).

## Compatibility baseline

| Component | Pin |
| --- | --- |
| CPython | 3.11 only (`requires-python = ">=3.11,<3.12"`) |
| Rextio | `>=0.1.3,<0.2` (plugin API **1.3**) |
| PyTorch | `torch==2.11.0` |
| Rust binding | `tch==0.24.0` with feature `python-extension` |
| Device | **CPU only** |

The published `tch` 0.24.0 release targets libtorch / PyTorch **2.11.0**
exactly. `LIBTORCH_BYPASS_VERSION_CHECK` is **not** an accepted build path.
Native builds that link the Python PyTorch install set `LIBTORCH_USE_PYTORCH=1`
and must use the same active CPython 3.11 environment as Rextio / PyO3.

## Phase A surface (certified)

One end-to-end route for float32 CPU tensors:

```python
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch
import torch.nn.functional as F

def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
```

The generated Rust path:

1. unpacks the three Python tensors through tch's python-extension bridge
   (another ref-counted handle; no storage copy);
2. runs functional linear, ReLU, and the literal-dimension mean under
   `no_grad` using fallible `tch` APIs;
3. keeps intermediate tensors as plugin-owned `RxtTorchTensor` values in Rust;
4. wraps the final tensor back into a Python `torch.Tensor`;
5. does not mutate inputs and does not build an autograd graph.

Runtime boundary extraction fail-closes on dtype/device/rank mismatches with
stable `rextio-torch:` messages (for example float64 where float32 is promised,
or rank-1 where rank-2 is promised).

### Annotation vocabulary (`rextio_torch.types`)

| Annotation | dtype | device | rank |
| --- | --- | --- | --- |
| `TensorF32Cpu2D` | float32 | CPU | 2 |
| `TensorF32Cpu1D` | float32 | CPU | 1 |

Marker classes intentionally **import neither torch nor Rextio**. Runtime
values remain ordinary `torch.Tensor` objects; the analyzer resolves the dotted
spellings when the plugin is enabled.

### Explicit exclusions (still fallback)

CUDA/MPS, training/autograd, arbitrary `nn.Module`, dynamic dtype/device/rank
or reduction dimensions, in-place ops, unsupported broadcasting, and the rest
of the broader 0.1.0 stretch surface are **not** claimed yet.

## Install (development only)

```bash
# CPython 3.11 only; requires rextio 0.1.3+ and torch 2.11.0
python3.11 -m pip install -e '.[dev]'
```

Plugin discovery and `rextio_torch.types` import **without** importing torch.
Real native builds need a matching torch 2.11 / libtorch environment and a Rust
toolchain. The `dev` extra includes packaging tools (`build`, `twine`,
`check-wheel-contents`) for the pre-release package gate.

## Residual platform risks

- `torch-sys` discovers Python via `PATH` / `VIRTUAL_ENV`, not only
  `PYO3_PYTHON`; a mismatched torch on `PATH` fails the exact 2.11.0 check.
- Cold builds compile `tch` / `torch-sys` for each generated project.
- The native extension must load under the same CPython 3.11 + torch 2.11.0
  that built it.
- Certification was executed on **macOS arm64**; other OS/arch combinations are
  not yet proven.
- Phase A remains private / unreleased; do not publish without a release review.

## Repository safeguards

- Do not publish to PyPI or change GitHub visibility during incubation.
- Do not use `LIBTORCH_BYPASS_VERSION_CHECK`.
- Do not add a project-local `AGENTS.md` without owner direction.
