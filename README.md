# rextio-torch

**Public Alpha** Rextio plugin that lowers a proven subset of Python
PyTorch inference code to Rust expressions backed by
[`tch`](https://github.com/LaurentMazare/tch-rs).

| Field | Value |
| --- | --- |
| Package version | **0.1.2** (from `rextio_torch.__about__`) |
| Release status | **Released 0.1.2 public Alpha** on **2026-07-26**; CPU reinforcement is released while CUDA E2 remains build-only and non-certifying |
| Distribution | [`rextio-torch==0.1.2`](https://pypi.org/project/rextio-torch/0.1.2/) on PyPI |
| Plugin API | **1.6** (`REQUIRED_PLUGIN_API`) |
| Product mode | Proven CPU inference plus a **Linux x86_64 build-only CUDA candidate**; no CUDA support claim |
| Certified host | **macOS arm64** (Apple Silicon), CPython **3.11**, torch **2.11.0** |
| Experimental hosts | **Linux x86_64**, **Linux AArch64** — runtime-backed but **not** certified |
| Availability-gated | **macOS x86_64** — pinned torch 2.11.0 CPython 3.11 wheel absent |
| Unsupported | Linux/macOS **i686** and **ARMv7** — no pinned runtime; impossible modern macOS targets |
| Deferred | **Windows** — unverified; no support claim |

This README documents the **0.1.2 compatibility,
classification-head, bounded CPU-surface update, build-only CUDA E2 candidate,
and opt-in real-NVIDIA execution-evidence candidate** to the
0.1.0 public native-AOT Alpha support contract. Every
form, rank, pin, and fail-closed behavior below is backed by current claim /
lower / rules / rust_snippets sources and the focused or real-Cargo tests that
exercise them. Unsupported sites must stay on the ordinary Python fallback or
be explicitly rejected — **never falsely claimed**.

Performance numbers under `benchmarks/results/` and
`benchmarks/results_phase_b/` are **historical context only**. They are **not**
a release gate for this Alpha cut.

---

## Version and ABI contract

| Component | Exact pin | Why it must match |
| --- | --- | --- |
| CPython | **3.11 only** (`requires-python = ">=3.11,<3.12"`) | PyO3 extension ABI and the dedicated certification venv; other CPython minor versions are not in the package contract. |
| Rextio package | **`>=0.1.6,<0.2`** | Required for structured device metadata and exact lowering authorization. |
| Plugin API | **1.6** | Adds device-value metadata and lowering authorization to the existing claim/lower contracts. |
| PyTorch | **`torch==2.11.0`** | Same major/minor/patch as the libtorch that published `tch` 0.24.0 expects. |
| Rust crate | **`tch =0.24.0`** with feature **`python-extension`** | Emitted by `crate_dependencies()`; `python-extension` supplies `pyobject_unpack` / `pyobject_wrap` for zero-storage-copy boundaries. |
| Generated Rust crate | Edition **2021**, `rust-version = "1.83"`, PyO3 **0.29** | Inherited from the Rextio 0.1.6 generated Cargo manifest. This is an MSRV/API contract, not an exact rustc patch pin. |
| Certified Rust toolchain | `rustc 1.93.1`, `cargo 1.93.1` on `aarch64-apple-darwin` | The real-Cargo Alpha evidence was reproduced with this local toolchain. This repo has no `rust-toolchain.toml`. |
| libtorch linkage | **`LIBTORCH_USE_PYTORCH=1`** | Builds against the active Python torch install. |
| Version-check bypass | **Forbidden** | `LIBTORCH_BYPASS_VERSION_CHECK` is **not** an accepted build path (stripped in e2e env setup; never set). |
| Device | **CPU proven; CUDA device 0 build-only** | CPU types retain their old checks. CUDA marker types require already-resident `cuda:0` values and never lower transfers. |
| Dtype | **float32 only** | Boundary extract rejects non-float32 at runtime. |
| Mode | **Inference / no-grad** | Every op helper wraps `tch::no_grad_guard()`; certified native outputs have `requires_grad is False` even when inputs request grad. |

### Why tch / libtorch / PyTorch / CPython must be one matched set

1. **tch 0.24.0 ↔ libtorch / PyTorch 2.11.0** — the published `tch` release
   targets that libtorch line exactly. Linking a different torch version fails
   the exact version check unless bypassed (bypass is rejected here).
2. **`LIBTORCH_USE_PYTORCH=1`** — `torch-sys` discovers Python via
   `PATH` / `VIRTUAL_ENV` (not only `PYO3_PYTHON`) and reads that interpreter’s
   torch. A mismatched torch on `PATH` fails the **2.11.0** check.
3. **CPython 3.11** — the generated native extension must load under the **same**
   CPython 3.11 + torch 2.11.0 environment that built it (PyO3 ABI + tch
   python-extension bridge).
4. **Rextio 0.1.6+ / provider API 1.6** — structured device metadata and
   lowering authorization are not available on older plugin APIs. The provider
   declares API 1.6 while Core's loader handles compatible later 1.x hosts.
   Registration fails closed for API 1.5 and older, other majors, or malformed host
   versions before registration can reach fields unavailable on older Core.

Certification and real-Cargo tests configure this environment explicitly.
**Host OS is not a runtime claim gate in this plugin:** the source does not
reject Linux solely because it is not macOS. Pins (CPython, torch, tch,
API) and toolchain availability still apply; mismatched or unavailable
tooling must fail visibly (never via `LIBTORCH_BYPASS_VERSION_CHECK`).

---

## Host platforms and CI truth model

Architecture labels are normalized here: `x86` means **i686** (32-bit), `x64`
means **x86_64/AMD64**, ARM32 means **ARMv7**, and ARM64 means **AArch64**.
Every requested Linux/macOS cell is represented in
`rextio_torch.platforms`; a green static contract cell is not a native support
claim.

| OS / arch | Alpha status | Native evidence / required result |
| --- | --- | --- |
| macOS ARM64 | **Certified** | Real Cargo on `macos-15` for every PR/push, with CPython 3.11, torch 2.11.0, tch 0.24.0, and rustc/cargo 1.93.1. |
| Linux x64 | **Experimental, runtime-backed** | Real Cargo on `ubuntu-24.04` for every PR/push with the same pins. Green is engineering evidence, not certification. |
| Linux ARM64 | **Experimental, runtime-backed** | Real Cargo on `ubuntu-24.04-arm` in scheduled/manual CI. It is deliberately non-blocking until hosted-runner evidence is reviewed. |
| macOS x64 | **Availability-gated, not supported** | The exact torch 2.11.0 CPython 3.11 wheel has no macOS x86_64 artifact. Scheduled/manual CI verifies that premise and requests reassessment if it changes. |
| Linux x86 / ARM32 | **Unsupported** | Static contract tests require a stable fail-closed result; torch 2.11.0 publishes no CPython 3.11 i686/ARMv7 wheel. No fake native job. |
| macOS x86 / ARM32 | **Unsupported / impossible modern target** | Static contract tests require a stable fail-closed result; neither a viable hosted runner nor pinned wheel exists. |
| Windows (all) | **Deferred** | Unverified and outside this Alpha train; no support claim. |

The blocking workflow separates `quality`, the eight-cell
`platform-contract`, runtime-backed `native-e2e`, and `package` jobs. Actions
receive only `contents: read`; third-party Actions are pinned to immutable
commit SHAs. A stable `CI gate` result aggregates every blocking lane for
branch protection. Native jobs build a clean wheel with pinned tooling, install
that wheel into a fresh venv, and reject any skipped runtime-backed test. The
scheduled/manual workflow owns expensive or
availability-gated experimental evidence.

Rules that carry `verified=True` mean the native rule family was compiled and
executed under the **certified** host contract above. They do **not** mean
every experimental Linux box is certified.

### Linux experimental verification recipe

Use this only to exercise the pinned environment on Linux. It does **not**
promote Linux to certified status.

```bash
# CPython 3.11 venv with the released source and exact Core evidence
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==26.1'
python -m pip install \
  'git+https://github.com/rextio/rextio.git@7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0' \
  'torch==2.11.0' 'pytest==9.1.1' 'ruff==0.15.22' 'mypy==2.3.0'
python -m pip install --no-deps -e .

# Required link mode; version-check bypass is forbidden
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK

# Fail closed if PATH does not resolve torch 2.11.0 for torch-sys
python -c "import torch; v=torch.__version__.split('+')[0]; assert v=='2.11.0', v"

# Focused unit tests (no native build)
pytest -q tests --ignore=tests/e2e

# Opt-in real-Cargo native slice (needs cargo + rustc; skips only if cargo missing)
# Failures from wrong torch / missing libtorch / ABI mismatch must surface — do not bypass.
pytest -q tests/e2e -m needs_cargo
```

Or run the maintainable wrapper (same contract, explicit env checks):

```bash
./scripts/linux-smoke.sh            # unit tests only
./scripts/linux-smoke.sh --cargo    # also run real-Cargo e2e when cargo is on PATH
```

If the pinned torch, CPython, Rextio API, or Cargo toolchain is wrong or
missing in a way that blocks the native path, the smoke must **fail or skip
with a clear reason** — never mask by setting `LIBTORCH_BYPASS_VERSION_CHECK`.

---

## Annotation vocabulary (`rextio_torch.types`)

| Annotation | Plugin type key | dtype | device | rank |
| --- | --- | --- | --- | --- |
| `TensorF32Cpu2D` | `rextio-torch/tensor-f32-cpu-2d` | float32 | CPU | 2 |
| `TensorF32Cpu1D` | `rextio-torch/tensor-f32-cpu-1d` | float32 | CPU | 1 |
| `TensorI64Cpu1D` | `rextio-torch/tensor-i64-cpu-1d` | int64 | CPU | 1 (classification result only) |
| `TensorF32Cuda0_2D` | `rextio-torch/tensor-f32-cuda0-2d` | float32 | CUDA device 0 | 2 (build-only) |
| `TensorF32Cuda0_1D` | `rextio-torch/tensor-f32-cuda0-1d` | float32 | CUDA device 0 | 1 (build-only) |

Marker classes intentionally **import neither torch nor Rextio**. Runtime values
remain ordinary `torch.Tensor` objects; the analyzer resolves the dotted
spellings when the plugin is enabled.

**What annotations prove:** dtype, device, and rank only.

**What they do not prove:** concrete dimensions. Matmul inner-size compatibility
and concrete PyTorch broadcasting are validated by **libtorch via fallible tch
APIs at execution time**.

Native Rust representation: plugin-owned `RxtTorchTensor(tch::Tensor)`.
Boundary unpack/wrap uses tch’s python-extension bridge (another ref-counted
handle, **no storage copy**). `Clone` uses `shallow_clone()` — in-place ops are
excluded because shallow clones alias storage.

### Build-only CUDA E2 candidate

The CUDA candidate is frozen to Linux x86_64, CPython 3.11, PyTorch/libtorch
2.11.0, tch 0.24.0, `LIBTORCH_USE_PYTORCH=1`, float32 rank-1/rank-2
`cuda:0`, and inference/no-grad. It accepts only named-intermediate
`rank2 @ rank2 -> rank2 + rank1 bias -> rank2.relu() ->
rank2.mean(dim=1) -> rank1`. Lowering requires the exact
`rextio-device-cuda/cuda-libtorch-linux-x86_64` authorization.

This is `support_claim=false`. CI runs actual Core orchestration with the real
provider implementation fed by a fixed synthetic probe report, then compiles
but never loads or runs the generated CUDA Rust slice. It does not inspect
host hardware or prove real NVIDIA execution, one-libtorch-image/ABI identity,
numerical equivalence, stream behavior, or performance. `.cuda()`/`.to()`,
CPU/CUDA mixing, reverse/same-rank add, mixed-rank CUDA matmul, functional CUDA
spellings, other devices, training/autograd, Windows, and macOS remain
excluded. See [the complete frozen CUDA contract](docs/cuda-build-only-0.1.2.md).

CUDA input and output boundaries also fail closed on non-strided layouts.
Because tch 0.24 has no general layout getter, native checks use its exposed
`is_sparse()` / `is_mkldnn()` flags and both Python boundaries additionally
require the pinned PyTorch 2.11.0 `tensor.layout` to render as
`torch.strided`; this catches sparse CSR/CSC/BSR/BSC without claiming a
version-independent tch layout query.

A separate manual Linux x86_64/NVIDIA harness executes the same exact slice
on a trusted maintainer host. It uses the real provider probe, compares
contiguous and noncontiguous values with eager PyTorch, requires
non-default-stream CUDA Graph replay and expected ATen CUDA activity, rejects
observed host/device transfers, and binds canonical wheel-relative
libtorch/c10 identities into offline-verifiable evidence. It remains
`support_claim=false` and `certification_ready=false`, is excluded from normal
CI, and does not expand the accepted operation or platform surface. See the
[CUDA candidate contract and commands](docs/cuda-build-only-0.1.2.md).

---

## Exactly supported Python forms and result ranks

Claims require API 1.6 metadata that proves the form (operand / receiver type
keys, call target, static keyword literals). Result ranks are those of the
claimed `result_type`.

| Form | Syntax contract | Operand / receiver ranks | Result rank | Rule id (native) |
| --- | --- | --- | --- | --- |
| Functional linear (tensor bias) | `torch.nn.functional.linear(x, w, b)` — three positional tensors | x, w: **2**; b: **1** | **2** | `rextio-torch/functional-linear-f32-cpu-2d` |
| Functional linear (no bias) | `torch.nn.functional.linear(x, w)`, `(x, w, None)`, or `(x, w, bias=None)`; x/w stay positional | x, w: **2** | **2** | `rextio-torch/functional-linear-none-f32-cpu-2d` |
| ReLU method | `.relu()` — zero args, no keywords | receiver **1** or **2** | **same as receiver** | `…/tensor-relu-f32-cpu-1d` or `…/tensor-relu-f32-cpu-2d` |
| Sigmoid method | `.sigmoid()` — zero args, no keywords | receiver **1** or **2** | **same as receiver** | `…/tensor-sigmoid-f32-cpu-1d` or `…/tensor-sigmoid-f32-cpu-2d` |
| Tanh method | `.tanh()` — zero args, no keywords | receiver **1** or **2** | **same as receiver** | `…/tensor-tanh-f32-cpu-1d` or `…/tensor-tanh-f32-cpu-2d` |
| Functional activations | Exact `torch.relu(t)`, `torch.sigmoid(t)`, or `torch.tanh(t)`; one positional tensor, no keywords; exact `torch.nn.functional.relu(t)` additionally permits omitted or literal `inplace=False` | input **1** or **2** | **same as input** | `…/function-{relu,sigmoid,tanh}-f32-cpu-rank1-2`; `…/functional-relu-f32-cpu-rank1-2` |
| Unary math | Exact `torch.{abs,neg,negative,square,exp,log,sqrt}(t)` or matching zero-argument tensor method; one tensor and no keywords | input/receiver **1** or **2** | **same as input** | `…/unary-{abs,neg,negative,square,exp,log,sqrt}-f32-cpu-rank1-2` |
| Exact GELU | `torch.nn.functional.gelu(t)` or `gelu(t, approximate="none")`; one positional tensor and no other options | input **1** or **2** | **same as input** | `…/functional-gelu-none-f32-cpu-rank1-2` |
| Matmul binop | `a @ b` | **2/2**, **2/1**, or **1/2** | **2** for 2/2; **1** for mixed ranks | `…/tensor-matmul-f32-cpu-{2d,mixed-rank}` |
| Matmul call | `torch.matmul(a, b)` — two positional; no keywords | **2/2**, **2/1**, or **1/2** | **2** for 2/2; **1** for mixed ranks | `…/tensor-matmul-call-f32-cpu-{2d,mixed-rank}` |
| Matmul method | `a.matmul(b)` — one positional; no keywords | receiver/other **2/2**, **2/1**, or **1/2** | **2** for 2/2; **1** for mixed ranks | same as call form |
| Elementwise `+` (same rank) | `a + b` | both **1**, or both **2** | **same as left** | `rextio-torch/tensor-add-f32-cpu-same-rank` |
| Elementwise `+` (bias broadcast) | `a + b` | one **2**, one **1** (either order) | **2** | `rextio-torch/tensor-add-f32-cpu-2d-1d-broadcast` |
| Elementwise `*` (same rank) | `a * b` | both **1**, or both **2** | **same as left** | `rextio-torch/tensor-mul-f32-cpu-same-rank` |
| Elementwise `*` (bias broadcast) | `a * b` | one **2**, one **1** (either order) | **2** | `rextio-torch/tensor-mul-f32-cpu-2d-1d-broadcast` |
| Elementwise `-` | `a - b`, same-rank or rank-2/rank-1 trailing broadcast in either operand order | input ranks **1/1**, **2/2**, **2/1**, or **1/2** | same rank, or **2** for mixed ranks | `…/tensor-sub-f32-cpu-{same-rank,2d-1d-broadcast}` |
| Elementwise true `/` | `a / b`, with the same rank/broadcast matrix as subtraction; binop only | input ranks **1/1**, **2/2**, **2/1**, or **1/2** | same rank, or **2** for mixed ranks | `…/tensor-div-f32-cpu-{same-rank,2d-1d-broadcast}` |
| Functional elementwise arithmetic | Exact `torch.add/sub/mul/div(a, b)`; two positional tensors and no keywords, using the same rank/broadcast matrix as the operators | input ranks **1/1**, **2/2**, **2/1**, or **1/2** | same rank, or **2** for mixed ranks | `…/function-{add,sub,mul,div}-f32-cpu-rank1-2` |
| Mean / sum | Receiver methods or exact `torch.mean` / `torch.sum`; `dim=0|1` once (positional literal or keyword); `keepdim` omitted=False or named bool | rank-2: dim 0/1; rank-1: only dim 0 + keepdim=True | registered float32 rank **1** or **2** only | legacy dim1 rules or `…/{mean,sum}-static-dim-f32-cpu-rank1-2` |
| Softmax | Receiver method or exact `torch.softmax`; exact `torch.nn.functional.softmax` additionally permits omitted or literal `dtype=None`; literal dim once | rank-1 dim 0; rank-2 dim 0/1 | same as input | legacy dim1 rule, `…/softmax-static-dim-f32-cpu-rank1-2`, or `…/functional-softmax-f32-cpu-rank1-2` |
| Argmax | Receiver method or exact `torch.argmax`; literal dim once; keepdim omitted=False or named bool | rank-2 dim 0/1 + keepdim=False; rank-1 dim 0 + keepdim=True | **1 int64** | legacy dim1 rule or `…/argmax-static-dim-i64-cpu-rank1` |

Native op helpers (all fallible, all under `no_grad`):

| Python form | Rust helper | tch API used |
| --- | --- | --- |
| linear | `__rxttorch_linear` / `__rxttorch_linear_no_bias` | `f_linear(Some(bias))` / `f_linear(None)` |
| activation method/function | `__rxttorch_{relu,sigmoid,tanh}` | fallible matching tch activation |
| unary math method/function | `__rxttorch_{abs,neg,negative,square,exp,log,sqrt}` | exact fallible `f_abs`, `f_neg`, `f_negative`, `f_square`, `f_exp`, `f_log`, or `f_sqrt` |
| exact GELU | `__rxttorch_gelu_none` | fixed fallible `f_gelu("none")`; the source option is never interpolated |
| mean / sum | `__rxttorch_{mean,sum}_dim{0,1}_keepdim_{true,false}` | `f_mean_dim` / `f_sum_dim_intlist` with fixed literals |
| `+` | `__rxttorch_add` | `f_add` |
| `*` | `__rxttorch_mul` | `f_mul` |
| `-` | `__rxttorch_sub` | `f_sub` |
| `/` | `__rxttorch_div` | `f_div` |
| matmul / `@` | `__rxttorch_matmul` | `f_matmul` |
| softmax | `__rxttorch_softmax_dim{0,1}` | `f_softmax` with fixed dim |
| argmax | `__rxttorch_argmax_dim{0,1}_keepdim_{true,false}` (only rank-1 outputs) | `f_argmax` plus exact CPU/int64/rank-1 check |

Coverage symbols declared for the analyzer include
`torch.nn.functional.{linear,gelu,relu,softmax}`, `torch.matmul`,
`torch.{relu,sigmoid,tanh,abs,neg,negative,square,exp,log,sqrt,add,sub,mul,div,mean,sum,softmax,argmax}`,
and the corresponding receiver methods. Binary `+` / `*` / `-` / `/` / `@`
are also claimed via binop sites. Functional options that change or widen
semantics, such as `alpha`, `out`, or `rounding_mode`, remain rejected.

### Control flow around claimed ops

Python `for` / `if` with Rextio-lowerable scalar `int` / `bool` conditions become
Rust control flow around the native tch helpers (certified control-flow
vertical slice). Tensor comparisons and tensor-data-dependent conditions are
**not** claimable by the current plugin API 1.6 surface and remain on the Python fallback.

**Core receiver limitation (not a plugin gap):** Rextio core does not offer
method calls whose receiver is a **bare BinOp** to plugins. Write named temps:

```python
# Accepted pattern
even = hidden @ weight + bias
hidden = even.relu()

# Not offered to plugins (bare BinOp receiver)
# hidden = (hidden @ weight + bias).relu()
```

Use **distinct temp names per if-arm** when both branches need intermediates
(core scopes Rust `let` bindings per arm).

---

## Static preconditions (analysis / claim time)

A site is claimed only when **all** of the following hold for that form.
Otherwise the plugin returns `Rejected` (recognized surface, wrong shape) or
`NotCovered` (unrelated / unresolved) — see [Compile-time fallback vs runtime
fail-closed](#compile-time-fallback-vs-runtime-fail-closed).

| Check | Enforcement |
| --- | --- |
| Plugin type keys are the registered float32 CPU rank-1/2 keys | `is_tensor_type` / exact type equality per rule |
| Linear: exact `torch.nn.functional.linear`; three positional tensors (2,2,1), or rank-2 input/weight with bias omitted / positional literal `None` / keyword literal `bias=None` | `claim/linear.py`; tensor-valued keywords remain unrepresentable in the Core API 1.6 ClaimSite contract |
| Activations: receiver zero-arg methods, or exact `torch.relu/sigmoid/tanh` with one positional tensor; exact `torch.nn.functional.relu` also permits omitted/literal `inplace=False`; rank 1/2 only | `claim/activations.py` |
| Unary math: receiver zero-arg methods, or exact `torch.abs/neg/negative/square/exp/log/sqrt` with one positional tensor; rank 1/2 and no keywords | `claim/unary.py` |
| GELU: exact `torch.nn.functional.gelu`; one positional rank-1/2 tensor; `approximate` omitted or exactly the static string literal `"none"`; lower always hardcodes `"none"` | `claim/gelu.py` |
| Reductions: receiver or exact `torch.mean/sum`; dim literal 0/1 once positionally/keyword; keepdim omitted=False or named bool; output must already map to rank-1/2 | `claim/reductions.py` |
| Classification: receiver or exact `torch.softmax/argmax`; exact `torch.nn.functional.softmax` also permits omitted/literal `dtype=None`; positional dim literal alignment and options are revalidated; argmax must map exactly to `TensorI64Cpu1D` | `claim/classification.py` |
| Matmul `@` / `torch.matmul` / `.matmul`: rank-2/rank-2 or one rank-2 plus one rank-1 operand in either order; call forms disallow keywords; method form takes one positional operand; rank-1/rank-1 stays rejected because it would return rank-0 | `claim/binops.py` |
| Elementwise arithmetic: binary `+` / `*` / `-` / `/`, or exact `torch.add/sub/mul/div(a, b)` with two positional non-literal tensors and no keywords; same-rank 1/1 or 2/2, or {1,2} broadcast; scalar/method/option variants and other ranks rejected | `claim/binops.py` |
| Claim metadata is pure function of site kind, target, operand types, receiver, static keyword literals | `claim/__init__.py` (config unused) |

Keyword order for `dim` / `keepdim` does not matter. A positional dim must be
an aligned non-bool int literal and is compile-time metadata, never a runtime
tensor helper operand. A second positional keepdim is excluded; keepdim is
omitted (real default `False`) or supplied as a named bool literal.

Lowering **independently revalidates** the same metadata (`rule_id`, ranks,
operand counts, receiver) and raises `ValueError` on drift — guards use
exceptions, not `assert`, so they survive `python -O`.

---

## Unsupported / fallback forms

These stay **unclaimed**, **rejected**, or **out of scope**. They must never
become silent native claims.

| Category | Examples / notes | Claim outcome (when offered as a torch site) |
| --- | --- | --- |
| Other devices | CUDA other than the exact build-only `cuda:0` slice; MPS and other accelerators | Outside the accepted vocabulary/slice; mixed devices and transfers are rejected |
| Other dtypes | float64, int, etc. | Outside vocabulary; boundary rejects non-float32 at runtime on native paths |
| Other ranks | rank-0 / rank-3+, rank-1 × rank-1 matmul, linear with wrong ranks | `Rejected` when form is recognized with wrong ranks |
| Training / autograd | backward, optimizers, parameter mutation | Out of scope; native helpers always `no_grad` |
| Modules | arbitrary `nn.Module` capture/execution | Uncovered / not claimed |
| Linear variants | tensor-valued keyword bias/input/weight, other keywords, method linear | Tensor keywords are not offered by the Core API 1.6 ClaimSite contract; others reject/fallback |
| In-place ops | `relu_`, `sigmoid_`, `tanh_`, unary `_` variants, in-place operators; `F.relu(..., inplace=True)` | Not claimed (zero-arg out-of-place methods only; `F.relu` admits only omitted/literal `inplace=False`) |
| Unary variants | scalar inputs; `out`; alternate aliases such as `torch.absolute`; method arguments or keywords | Rejected for recognized exact functions/methods or otherwise unclaimed |
| GELU variants | `approximate="tanh"`, dynamic/invalid/positional approximate, other keywords, `.gelu()`, or `nn.GELU` capture | Static recognized variants are rejected; dynamic values may be filtered by Core to ordinary fallback before plugin claim |
| Elementwise variants | scalar operands; `.add/.sub/.mul/.div`; aliases such as `torch.subtract`; `alpha`, `out`, or `rounding_mode` options | Rejected for recognized exact functions/options or otherwise unclaimed; `/` and exact `torch.div(a, b)` mean tensor-tensor true division only |
| Reductions other shapes | whole-tensor reduction, dynamic/duplicate/out-of-range dim, positional keepdim, dtype/out, or rank-0 result | `Rejected` or unclaimed |
| Classification variations | dtype override, dynamic/duplicate dim, softmax keepdim, rank-2 argmax keepdim=True, rank-1 argmax keepdim=False, rank-3+ | `Rejected` or unclaimed; unavailable int64 rank-2/rank-0 types are not invented |
| Classification-result arithmetic | `TensorI64Cpu1D` or mixed int64/float32 operands with `+`, `*`, `-`, or `/` | `Rejected`; elementwise arithmetic remains float32-only |
| Views / reshape | transpose, view, reshape (alias / shallow-clone risk) | Intentionally not claimed |
| Unsupported broadcast ranks | arithmetic rank combinations other than same-rank or 2d/1d | `Rejected` |
| Unrelated torch APIs | e.g. `torch.subtract` / `torch.divide` aliases | `NotCovered` |
| Unresolved types | missing annotation / `None` operand types | `NotCovered` (no false claim) |
| Tensor-dependent control flow | tensor comparisons as `if` conditions | Not claimable under the current API 1.6 surface |
| Bare BinOp method receivers | `(a @ b).relu()` | Core does not offer site; use temps |
| Version bypass | `LIBTORCH_BYPASS_VERSION_CHECK` | Forbidden build path |
| Custom operators | user ops outside the table above | Uncovered |

Rule record `rextio-torch/unsupported-tensor-surface` (`RXTP-TORCH-010`,
outcome **fallback**) documents the fail-closed rejection lane for covered torch
sites whose types or shapes fall outside the Alpha surface.

---

## Compile-time fallback vs runtime fail-closed

Two different layers. Do not conflate them.

### Analysis / codegen time (before or while building native code)

| Outcome | Meaning | User effect |
| --- | --- | --- |
| **Claimed** | Metadata proves a native rule | Site can lower to a tch helper |
| **Rejected** | Recognized surface, wrong static shape/types | Diagnostic error (`RXTP-TORCH-*`); site stays on **Python fallback** — never a false native claim |
| **NotCovered** | Not this plugin’s target, or types unresolved | Other plugins / ordinary Python fallback |
| **Lower `ValueError`** | Claimed metadata changed or is inconsistent at lower | Codegen fails closed (no `assert` / no silent bad emit) |

Native rules that passed real-Cargo certification are marked `verified=True` in
`rules/records.py`. The unsupported-surface fallback record is not a native
op and remains `verified=False`.

### Runtime (native extension executing)

| Check | Failure behavior | Typical message / path |
| --- | --- | --- |
| Boundary: value is a `torch.Tensor` | Python exception | `rextio-torch: expected a torch.Tensor` |
| Boundary: device is CPU | `ValueError` | `rextio-torch: expected a CPU tensor` |
| Boundary: dtype is float32 | `ValueError` | `rextio-torch: expected a float32 tensor` |
| Boundary: rank matches annotation | `ValueError` | `rextio-torch: expected rank-N tensor, got rank M` |
| Matmul inner dimensions | Fallible `f_matmul` → mapped error | `TchError` → `PyRuntimeError` via `__rxttorch_map_err` |
| Add / multiply broadcast concrete sizes | Fallible `f_add` / `f_mul` | same mapping |
| Other op failures | Fallible `f_*` APIs under `no_grad` | same mapping; no `unwrap` / `panic!` / `assert` |

Unary domain values are normal tensor results, not operation failures:
`log`/`sqrt` can produce NaN, zero can map to signed infinity, and overflow can
produce infinity. Native certification compares eager/native NaN and infinity
classes, finite values, and signed zero under the pinned backend. NaN payload
bits and NaN sign are intentionally not portable cross-platform guarantees.
Exact GELU uses the same comparison policy (`-∞` produces NaN, `+∞` remains
infinite, and signed zero is preserved).

Certified e2e tests force **native-only** mode for the boundary rejects they
exercise, so eager fallback cannot mask float64/wrong-rank failures. Inputs are
not mutated; native outputs remain no-grad CPU float32 of the promised rank
when contracts hold. The non-CPU check and incompatible concrete
matmul/broadcast errors are source-enforced fallible paths, but the CPU-only
real-Cargo fixtures do not directly execute those failure cases.

---

## Examples of accepted vs rejected syntax

### Accepted (Phase A certified chain)

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

Claims: linear (2→2), relu (2→2), mean (2→1). Real-Cargo certified.

### Accepted (control-flow + expanded Alpha surface)

```python
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch

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

def expanded_surface(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    offset: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    hidden = torch.matmul(x, weight)
    hidden = hidden.tanh()
    hidden = hidden + hidden
    hidden = hidden.matmul(weight)
    reduced = hidden.sum(dim=1, keepdim=False)
    reduced = reduced + offset
    return reduced.relu().sigmoid().tanh()
```

One serialized certification project exercises every Alpha rule family (both
matmul call forms, same-rank and 2D+1D add families, sum, tanh, rank-1
activation chaining, and the control-flow route).

### Accepted (bounded 0.1.2 CPU follow-up)

```python
def cpu_surface_followup(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
    divisor: TensorF32Cpu1D,
) -> TensorF32Cpu2D:
    hidden = torch.nn.functional.linear(x, weight, bias=None)
    activated = torch.relu(hidden)
    shifted = torch.sigmoid(activated) - bias
    scaled = shifted / divisor
    totals = torch.sum(scaled, 0, keepdim=True)
    averaged = torch.mean(totals, 1, keepdim=True)
    probabilities = torch.softmax(averaged, 1)
    return torch.tanh(probabilities)
```

The same serialized real-Cargo project executes this slice plus omitted and
positional-`None` linear forms, every same-rank/2D↔1D subtraction and division
order, all six mixed-rank matmul spellings/order combinations, functional
rank-2 argmax with default `keepdim=False`, and rank-1 argmax with named
`keepdim=True`. It checks numerical parity, exact dtype/device/rank, no-grad
output, non-mutation, native route evidence, and runtime incompatible
matmul/broadcast failures. Direct unary kernels additionally certify NaN/Inf
domain behavior and signed-zero parity for every exact fallible helper; default
and explicit-`none` GELU are separately routed and compared.

### Rejected or not covered (illustrative)

```python
# Rejected: linear wrong ranks (1d input)
# F.linear(x_1d, weight_2d, bias_1d)

# Rejected: rank-1 reduction would produce unregistered rank-0
# vector.mean(dim=0)  # keepdim defaults to False

# Rejected: positional keepdim is deliberately not represented
# t.mean(0, True)

# Rejected: duplicate positional and keyword dim
# t.sum(0, dim=1)

# Rejected: rank-1 × rank-1 matmul would produce unregistered rank-0
# a_1d @ b_1d

# Rejected: exact i64 output type is unavailable
# matrix.argmax(dim=1, keepdim=True)  # would be int64 rank-2
# vector.argmax(dim=0)                # would be int64 rank-0

# Rejected: semantic-changing functional option
# torch.div(a, b, rounding_mode="trunc")

# Rejected: only exact GELU approximation "none" is native
# torch.nn.functional.gelu(x, approximate="tanh")

# Not offered by the current Core API 1.6 ClaimSite contract: runtime tensor-valued keyword
# F.linear(x, weight, bias=bias_tensor)

# Not offered by core to plugins: bare BinOp receiver
# (a @ b + bias).relu()

# Not claimed: in-place / unsupported elementwise forms / views
# x.relu_();  x * 2.0;  x.transpose(0, 1)

# Accepted classification head: int64 rank-1 labels
# logits.softmax(dim=1).argmax(dim=1, keepdim=False)
```

---

## Install

The public PyPI command installs the released **0.1.2 / plugin API 1.6
Alpha**. Its CPU surface is released; the included CUDA lane remains the
explicitly build-only, non-certifying candidate documented below:

```bash
python3.11 -m pip install 'rextio-torch==0.1.2'
```

To reproduce the reviewed **0.1.2 / plugin API 1.6** source evidence exactly,
install the pinned Core commit and this checkout without dependency resolution:

```bash
python3.11 -m pip install \
  'git+https://github.com/rextio/rextio.git@7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0' \
  'torch==2.11.0' 'pytest==9.1.1' 'ruff==0.15.22' 'mypy==2.3.0'
python3.11 -m pip install --no-deps -e .
```

The reviewed 0.1.2 evidence pins are Core
`7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0` and CUDA provider
`a5fb427e91710b65f54ee5b8e33706c45840cf9c`. The provider is not a mandatory
package dependency; install it only for the maintainer CUDA orchestration/build
gate:

```bash
python3.11 -m pip install --no-deps \
  'git+https://github.com/rextio/rextio-device-cuda.git@a5fb427e91710b65f54ee5b8e33706c45840cf9c'
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK
python3.11 scripts/build_cuda_candidate.py \
  --output /tmp/rextio-torch-cuda-build-only
```

Plugin discovery and `rextio_torch.types` import **without** importing torch.
Real native builds need a matching torch 2.11 / libtorch environment and a Rust
toolchain, with:

```bash
export LIBTORCH_USE_PYTORCH=1
# never: LIBTORCH_BYPASS_VERSION_CHECK
# ensure `python` on PATH is the same CPython 3.11 env as PyO3 / Rextio
```

The `dev` extra includes packaging tools (`build`, `twine`,
`check-wheel-contents`) for release artifact checks.

---

## Historical benchmarks (not a release gate)

Phase A and Phase B product-route benchmarks were preregistered and executed
with fixed cells. Recorded numbers live under:

- `benchmarks/results/` (Phase A)
- `benchmarks/results_phase_b/` (Phase B deep control)

Those runs measured boundary-inclusive latency and expansion GO criteria. For
this Alpha cut they are **historical evidence only** — usefulness of the pinned
AOT surface is the product goal, not beating a speedup threshold.

---

## Residual platform risks

- `torch-sys` discovers Python via `PATH` / `VIRTUAL_ENV`, not only
  `PYO3_PYTHON`; a mismatched torch on `PATH` fails the exact 2.11.0 check.
- Cold builds compile `tch` / `torch-sys` for each generated project.
- The native extension must load under the same CPython 3.11 + torch 2.11.0
  that built it.
- **Certified** real-Cargo evidence remains **macOS arm64 only**. Linux
  x86_64/AArch64 are **experimental** runtime-backed profiles; results there do
  not rewrite certification without a deliberate re-record.
- The eight-cell platform table is declarative and never probes or silently
  rejects the ambient host. Native CI calls its explicit runtime requirement
  before building; unavailable cells fail closed with stable reason codes.
- Linux residual risks include distro libstdc++/glibc differences, torch wheel
  manylinux tags, and first-time `tch` compile cost — not silent OS rejection
  by this plugin.
- **Windows** is **deferred** (unverified).
- Core limitation: method claims require named or call-chain receivers, not
  bare BinOp receivers.
- Release artifacts are built from clean `main` and checked independently from
  the historical benchmark evidence.

---

## Repository safeguards

- The temporary `Private` upload-block classifier was removed after
  supported-profile CI and clean merged-`main` review.
- The annotated tag, PyPI artifacts, and live no-cache install are verified as
  separate release records rather than inferred from repository metadata.
- Do not use `LIBTORCH_BYPASS_VERSION_CHECK`.
- Do not add a project-local `AGENTS.md` without owner direction.
- Package metadata identifies released version **0.1.2** as Development Status Alpha.

For the longer product definition and phase history, see the
[0.1.0 implementation plan](docs/implementation-plan-0.1.0.md). Historical
benchmark protocols are indexed in [benchmarks/README.md](benchmarks/README.md);
this README is the current support surface.
