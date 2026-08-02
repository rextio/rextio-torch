# rextio-torch

<p align="center">
  <img src="https://raw.githubusercontent.com/rextio/rextio-torch/main/assets/readme/rextio-icon.png" width="112" alt="Rextio project icon">
</p>

<p align="center"><strong>Bounded PyTorch inference lowering from Python to Rust-backed <code>tch</code> operations.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/rextio-torch/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-torch?label=PyPI" alt="rextio-torch on PyPI"></a>
  <a href="https://github.com/rextio/rextio-torch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

<p align="center">
  <strong>English</strong> · <a href="https://github.com/rextio/rextio-torch/blob/main/README.ko.md">한국어</a> · <a href="https://github.com/rextio/rextio-torch/blob/main/README.zh-hans.md">简体中文</a> · <a href="https://github.com/rextio/rextio-torch/blob/main/README.zh-hant.md">繁體中文</a> · <a href="https://github.com/rextio/rextio-torch/blob/main/README.ja.md">日本語</a>
</p>

`rextio-torch` is a public Alpha plugin for [Rextio](https://github.com/rextio/rextio). It recognizes a deliberately small, proven PyTorch inference surface and emits fallible Rust expressions backed by [`tch`](https://github.com/LaurentMazare/tch-rs). Unsupported code remains on Rextio's ordinary Python fallback or is rejected with a diagnostic; it is never silently claimed as native.

> [!IMPORTANT]
> This is an inference-only, CPU-first Alpha. It does not support training, autograd, optimizers, arbitrary `nn.Module` execution, or in-place tensor operations. The CUDA path is build-only and remains `support_claim=false` and `certification_ready=false`.

## See the bounded path

This function is inside the certified CPU surface:

```python
import torch.nn.functional as F
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D


def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
```

Rextio can prove the rank/type route for `linear → relu → mean`, lower it through this plugin, and execute fallible `tch` helpers. Concrete matrix and broadcast compatibility is still validated by libtorch at runtime.

There is no performance promise. The retained Phase A and Phase B benchmarks are historical, workload-specific evidence; Phase B recorded a **NO-GO**, and the 0.1.3 small-batch harness is diagnostic only.

## How it works

```text
typed Python function
        ↓ Rextio analysis and plugin claim
fallible tch expression helpers
        ↓ PyO3 host extension
active PyTorch/libtorch 2.11.0 runtime
```

- Import-free marker types describe dtype, device, and rank—not concrete dimensions.
- Tensor boundaries use `tch`'s Python-extension bridge and another reference-counted handle; they do not copy tensor storage.
- Eligible PyO3 functions on Core plugin API 1.7 install one function-scoped `tch::no_grad_guard()` after input conversion and drop it before output conversion.
- RXT075 Python-boundary functions, standalone/type-only paths, and legacy/no-hook contexts keep per-operation no-grad guards.
- Native outputs have `requires_grad is False`; RAII restores the caller's prior grad mode on success and error exits.

## Install and try

Use CPython 3.11 and the exact PyTorch line:

```bash
python3.11 -m pip install 'rextio>=0.1.7,<0.2' 'rextio-torch==0.1.3'
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK
```

The package registers itself through the `rextio.plugins` entry point. Put the example above in a Rextio project and use the normal Rextio analysis/build flow. Plugin discovery and `rextio_torch.types` do not import PyTorch; a native build does require the matching active PyTorch/libtorch environment and a Rust toolchain.

`LIBTORCH_BYPASS_VERSION_CHECK` is not an accepted build path. The Python found through `PATH`/`VIRTUAL_ENV` must be the same CPython 3.11 environment used by PyO3 and must contain `torch==2.11.0`.

## Compatibility contract

| Component | Contract |
| --- | --- |
| Package | `rextio-torch==0.1.3` (public Alpha, released 2026-07-27) |
| CPython | `>=3.11,<3.12` — 3.11 only |
| Rextio | `>=0.1.7,<0.2` |
| Plugin API | `1.7` |
| PyTorch/libtorch | `torch==2.11.0` |
| Rust binding | `tch =0.24.0`, feature `python-extension` |
| Generated crate | Rust edition 2021, `rust-version = "1.83"`, PyO3 0.29 |
| Certified toolchain evidence | `rustc 1.93.1`, `cargo 1.93.1` on `aarch64-apple-darwin` |
| Required linkage | `LIBTORCH_USE_PYTORCH=1`; version-check bypass forbidden |
| CPU tensors | float32 rank 1 or 2; classification result may be int64 rank 1 |
| Mode | inference/no-grad only |

### Platform status

| Host | Status |
| --- | --- |
| macOS arm64 | **Certified Alpha** real-Cargo path |
| Linux x86_64 | **Experimental, runtime-backed**; not certified |
| Linux AArch64 | **Experimental, runtime-backed** manual/scheduled path; not certified |
| macOS x86_64 | Availability-gated and unsupported: no pinned torch 2.11.0 CPython 3.11 wheel |
| Linux/macOS i686 or ARMv7 | Unsupported |
| Windows | Deferred and unverified; no support claim |

## Supported CPU surface

All tensor operands are registered float32 CPU rank-1/rank-2 values unless the table says otherwise. Options must be static literals exactly as described.

| Family | Accepted forms and boundaries |
| --- | --- |
| Linear | Exact `torch.nn.functional.linear(x, w, b)` with ranks 2/2/1; or no bias via omission, positional `None`, or literal `bias=None` |
| Activations | `.relu()`, `.sigmoid()`, `.tanh()`; exact `torch.relu/sigmoid/tanh`; exact `F.relu` with omitted or literal `inplace=False`; rank 1/2 |
| Unary math | Exact `torch.abs/neg/negative/square/exp/log/sqrt` or matching zero-argument methods; rank 1/2 |
| GELU | Exact `torch.nn.functional.gelu(t)` with omitted or literal `approximate="none"` only |
| Matmul | `a @ b`, `torch.matmul(a, b)`, or `a.matmul(b)` for 2×2, 2×1, or 1×2; rank-1 × rank-1 is excluded |
| Elementwise | `+`, `*`, `-`, `/` and exact `torch.add/sub/mul/div(a, b)` for 1/1, 2/2, 2/1, or 1/2; tensor-tensor only and no semantic-changing options |
| Reductions | Method or exact `torch.mean/sum`; literal `dim=0|1`; `keepdim` omitted/false or named bool; only registered rank-1/2 results |
| Softmax | Method or exact `torch.softmax`; exact `F.softmax` also permits omitted/literal `dtype=None`; rank-1 dim 0 or rank-2 dim 0/1 |
| Argmax | Method or exact `torch.argmax`; rank-2 dim 0/1 with `keepdim=False`, or rank-1 dim 0 with `keepdim=True`; result is int64 rank 1 |
| Scalar control flow | Python `for`/`if` around claimed operations when Rextio proves scalar `int`/`bool` conditions |

Concrete dimension errors use fallible libtorch APIs and become Python exceptions. Unary `log`/`sqrt` domain behavior follows the pinned eager backend, including NaN/infinity and signed-zero behavior.

### Marker types

| Annotation | Meaning |
| --- | --- |
| `TensorF32Cpu2D` | float32, CPU, rank 2 |
| `TensorF32Cpu1D` | float32, CPU, rank 1 |
| `TensorI64Cpu1D` | int64, CPU, rank 1 classification result |
| `TensorF32Cuda0_2D` | float32, `cuda:0`, rank 2 — build-only |
| `TensorF32Cuda0_1D` | float32, `cuda:0`, rank 1 — build-only |

## What falls back or fails closed

The plugin does not claim other dtypes, rank 0 or rank 3+, other devices, transfers, arbitrary modules, mutation/in-place operations, views/reshape/transpose, scalars in elementwise operations, tensor-dependent branches, dynamic/duplicate dimensions, semantic-changing options, unregistered output ranks, or unrelated aliases.

Recognized but invalid static shapes/options are rejected with `RXTP-TORCH-*` guidance and stay on Python fallback. Unresolved or unrelated forms are `NotCovered`. Lowering revalidates claimed metadata and raises `ValueError` on drift. A native boundary rejects the wrong Python type, device, dtype, rank, or layout; fallible `tch` errors map to Python exceptions rather than `unwrap`, panic, or silent replay.

## CUDA: evidence, not support

The only CUDA candidate is frozen to Linux x86_64, CPython 3.11, PyTorch/libtorch 2.11.0, `tch` 0.24.0, float32 rank-1/rank-2 tensors already resident on `cuda:0`, and this named-intermediate slice:

```text
rank2 @ rank2 → rank2 + rank1 bias → rank2.relu() → rank2.mean(dim=1) → rank1
```

It requires authorization from `rextio-device-cuda/cuda-libtorch-linux-x86_64`. Hosted CI uses a synthetic probe and compiles the generated extension but never loads it or executes CUDA. A retained WSL2/RTX 3060 (`sm_86`) manual run produced verifier-success evidence, including observed kernel activity, but it remains opt-in evidence only:

```text
support_claim=false
certification_ready=false
```

No `.cuda()`/`.to()` lowering, transfers, CPU/CUDA mixing, other devices, multi-GPU, training, autograd, Windows/macOS CUDA, or CUDA performance claim is included. Read the [frozen CUDA contract](docs/cuda-build-only-0.1.2.md) before using the maintainer harness.

## Further detail

- [Function-scoped no-grad contract](docs/invocation-scope-proposal-0.1.3.md)
- [Diagnostic small-batch scoring protocol](docs/preregister-small-batch-scoring-diagnostic-0.1.3.md)
- [Historical benchmark index](benchmarks/README.md)
- [Changelog](CHANGELOG.md)

## License

[MIT](LICENSE)
