# CUDA E2 build-only candidate (unreleased 0.1.2)

This document defines the entire CUDA scope in this branch. It is an
engineering candidate, not a CUDA support claim. All provider reports remain
`support_claim=false`.

## Frozen environment

- Linux x86_64 only
- CPython 3.11
- PyTorch/libtorch 2.11.0
- `tch` 0.24.0 with `python-extension`
- `LIBTORCH_USE_PYTORCH=1`
- Rextio plugin API 1.6
- exact device provider `rextio-device-cuda`
- exact capability `cuda-libtorch-linux-x86_64`
- float32 rank-1/rank-2 tensors already resident on `cuda:0`
- inference/no-grad only

The reviewed cross-repository evidence inputs are:

- Core `rextio/rextio` commit
  `7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0`
- CUDA provider `rextio/rextio-device-cuda` commit
  `a5fb427e91710b65f54ee5b8e33706c45840cf9c`

The CUDA provider remains an optional maintainer build input, not a
`rextio-torch` package dependency.

Core canonicalizes `cuda:0` as logical device `gpu:0` plus backend `cuda`.
The plugin types also require the `libtorch` runtime, runtime reuse, strided
layout, device memory, inference/no-grad features, libtorch 2.11.0
(`cuda`, `pytorch-wheel`), and tch 0.24.0 (`cuda`). These are passive planning
facts. Lowering separately requires a matching Core-produced authorization and
checks both provider and capability IDs.

## Only accepted CUDA slice

Use named intermediates:

```python
from rextio_torch.types import TensorF32Cuda0_1D, TensorF32Cuda0_2D


def inference(
    x: TensorF32Cuda0_2D,
    weight: TensorF32Cuda0_2D,
    bias: TensorF32Cuda0_1D,
) -> TensorF32Cuda0_1D:
    hidden = x @ weight
    biased = hidden + bias
    activated = biased.relu()
    return activated.mean(dim=1)
```

The exact matrix is:

| Python form | Types | Result | Rust/tch |
| --- | --- | --- | --- |
| `x @ weight` | rank-2, rank-2 | rank-2 | `f_matmul` |
| `hidden + bias` | rank-2, rank-1, in that order | rank-2 | `f_add` |
| `biased.relu()` | rank-2 receiver | rank-2 | `f_relu` |
| `activated.mean(dim=1)` | rank-2 receiver, literal `1`, `keepdim=False` | rank-1 | `f_mean_dim` |

Every helper uses `no_grad_guard`. Boundary extraction verifies
`Device::Cuda(0)`, `Kind::Float`, exact rank, and strided layout.
Rank-specific materializers repeat those checks, then use `pyobject_wrap`;
there is no `.cpu()`, `.to()`, or storage copy.

`tch` 0.24 exposes `is_sparse()` and `is_mkldnn()` but no general tensor
layout getter (and no `is_sparse_csr()` wrapper). The native boundary rejects
the exposed sparse/MKLDNN flags, while both input and wrapped output also
require the pinned PyTorch 2.11.0 object's exact `layout` rendering to be
`torch.strided`. That pinned Python-boundary check is what also rejects
sparse CSR/CSC/BSR/BSC layouts; it is not claimed as a version-independent
tch layout API.

## Explicit exclusions

- `.cuda()` and `.to(...)` lowering or any host/device transfer
- CPU/CUDA mixing, reverse-order bias add, or same-rank CUDA add
- mixed-rank matmul, functional `torch.matmul`/`torch.relu`, and all broader
  CPU-only operations
- `cuda:1`, dynamic devices, multiple GPUs, mutation/in-place operations
- training, autograd, Windows CUDA, macOS CUDA, TensorFlow
- performance claims

The GitHub lane is deliberately GPU-free. It runs actual Core orchestration:
the analyzer accepts the annotated four-op fixture, Core derives the exact
DeviceRequirement/runtime rows, resolves the real installed
`CudaDeviceProvider`, binds its plan and lowering authorization, generates the
PyO3 Cargo project, and compiles that project once with `cargo build
--release`. The generated cdylib is never loaded and no CUDA function runs.

The provider implementation and preflight logic are real, but its probe runner
returns a fixed synthetic Linux x86_64 report (driver 12080, ordinal 0,
`sm_80`). It does not inspect the runner's driver, toolkit, or hardware.
Assertions retain `support_claim=false`, verify the borrow-only framework
contribution and profile/authorization hash binding, and inspect the generated
Cargo/Rust before compilation. This does not prove a CUDA driver, real NVIDIA
execution, numerical equivalence, current-stream behavior, or that Python and
generated Rust resolved one identical libtorch image.

## Maintainer GPU-free orchestration/build

After creating and activating a CPython 3.11 venv from this repository:

```bash
python -m pip install 'pip==26.1'
python -m pip install \
  'git+https://github.com/rextio/rextio.git@7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0' \
  'torch==2.11.0'
python -m pip install --no-deps \
  'git+https://github.com/rextio/rextio-device-cuda.git@a5fb427e91710b65f54ee5b8e33706c45840cf9c'
python -m pip install --no-deps -e .
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK
python scripts/build_cuda_candidate.py \
  --output /tmp/rextio-torch-cuda-build-only
```

This command is Linux x86_64-only. It uses the fixed synthetic probe described
above, links the generated cdylib, verifies that file exists, and never loads
or executes it.

## Manual real-NVIDIA execution candidate

`scripts/certify_cuda_candidate.py` is an explicit maintainer-run candidate,
never an ordinary GitHub-hosted CI job. It requires clean Core and provider
checkouts at the commits above, a clean `rextio-torch` candidate descending
from `7d6fb1b606bfab6530a7ff96317081b2fd1c1b22`, Rust 1.93.1, a CUDA-enabled
PyTorch 2.11.0 wheel, `ldd`, `readelf`, and NVIDIA device ordinal 0.

On a trusted Linux x86_64 NVIDIA host:

```bash
# From the rextio-torch checkout, with sibling exact Core/provider checkouts.
python -m pip install --no-deps -e ../rextio
python -m pip install --no-deps -e ../rextio-device-cuda
python -m pip install --no-deps -e .
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK
TORCH_COMMIT="$(git rev-parse HEAD)"
EVIDENCE_PATH="/tmp/rextio-torch-cuda-e2.json"
python scripts/certify_cuda_candidate.py \
  --expected-torch-commit "${TORCH_COMMIT}" \
  --sm sm_80 \
  --work-dir /tmp/rextio-torch-cuda-e2-work \
  --output "${EVIDENCE_PATH}"
python scripts/verify_cuda_e2_evidence.py \
  "${EVIDENCE_PATH}"
```

The editable local installs are mandatory for this manual command: the harness
checks that imported Core, provider, and Torch plugin modules resolve under the
three exact clean checkouts whose commits enter evidence. Installing those
commits into unrelated `site-packages` paths does not satisfy that identity
check.

Do not derive an `LD_LIBRARY_PATH` entry from
`torch.utils.cmake_prefix_path` or append `/lib` to that CMake path. It points
under `torch/share/cmake`, while the wheel DSOs are under the `lib` directory
beside the exact imported `torch.__file__`. The harness derives that wheel-local
directory itself and prepends it to the environment supplied to each `ldd`
subprocess; an existing ambient `LD_LIBRARY_PATH` is retained only after that
exact path.

Replace `sm_80` with the exact device SM: `sm_60`, `sm_61`, `sm_70`, `sm_72`,
`sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_89`, or `sm_90`. Use a new,
non-existing work directory for every run.

WSL2 remains part of the Linux x86_64 experimental/manual-evidence path only.
A successful WSL2 run is useful execution evidence, but it does not establish
a Windows or CUDA support claim and does not change `support_claim=false` or
`certification_ready=false`.

The harness builds the exact provider probe, routes its absolute path through
real Core preflight, generates/builds the four-op extension, imports PyTorch
first, and executes contiguous and noncontiguous tensors already on `cuda:0`.
It requires eager numerical equivalence, output/input/lifetime contracts,
non-default-stream CUDA Graph capture/replay, no static or profiled transfer,
and CUDA activity for expected matmul/add/ReLU/mean ATen operations.
`/proc/self/maps` and `ldd` must agree on canonical PyTorch-wheel
`libtorch*`/`libc10*` images.

The replay check copies new values into the captured static inputs on the
selected non-default stream before replay, so a same-input cached result cannot
pass. Successful inputs retain value, stride, storage offset, and data pointer;
grad-requesting inputs still produce no-grad output. CPU, float64, wrong-rank,
and sparse-layout inputs must fail at the native boundary. The Cargo build is
bound to the active CPython 3.11 virtual environment through `VIRTUAL_ENV`,
`PATH`, and `PYO3_PYTHON`, with a subprocess interpreter/torch identity check.

Evidence is canonical, size/depth-bounded JSON with a non-circular payload
hash. It retains wheel-relative paths, hashes, sizes, and ELF build IDs, but
not raw maps/`ldd`, machine paths, environment values, URLs, stream pointers,
or timings. The offline verifier rejects unknown fields, payload tampering,
weakened tolerances, incomplete invariants, and every support/certification
overclaim.

The payload also binds the exact ordered lowering-rule IDs, generated
`lib.rs`/`Cargo.toml`/`Cargo.lock`, built extension, immutable provider probe,
and equal authorization/provider-lock artifact-profile hashes. Installed
`direct_url.json` contributes only sanitized VCS commit, editable flag, or
safe archive-hash identity; URLs, paths, credentials, and requested revisions
are neither retained nor hashed.

A successful run still records:

```text
support_claim=false
certification_ready=false
kernel_executed=true
```

The last value is emitted only after expected ATen operations show CUDA
activity. It is real-execution evidence, not a release or support claim.
