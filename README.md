# rextio-torch

**Public Alpha** Rextio plugin that lowers a proven subset of Python
PyTorch inference code to Rust expressions backed by
[`tch`](https://github.com/LaurentMazare/tch-rs).

| Field | Value |
| --- | --- |
| Package version | **0.1.1** (unreleased; from `rextio_torch.__about__`) |
| Release status | **Unreleased compatibility hotfix**; latest release is **0.1.0 public Alpha** (2026-07-18) |
| Distribution | Latest published release: [`rextio-torch==0.1.0`](https://pypi.org/project/rextio-torch/0.1.0/) on PyPI |
| Plugin API | **1.3** (`REQUIRED_PLUGIN_API`) |
| Product mode | **CPU inference / no-grad only** |
| Certified host | **macOS arm64** (Apple Silicon), CPython **3.11**, torch **2.11.0** |
| Experimental hosts | **Linux x86_64**, **Linux AArch64** — runtime-backed but **not** certified |
| Availability-gated | **macOS x86_64** — pinned torch 2.11.0 CPython 3.11 wheel absent |
| Unsupported | Linux/macOS **i686** and **ARMv7** — no pinned runtime; impossible modern macOS targets |
| Deferred | **Windows** — unverified; no support claim |

This README documents the **unreleased 0.1.1 compatibility update** to the
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
| Rextio package | **`>=0.1.3,<0.2`** | Allowed package range, not an exact pin. Core's loader negotiates this provider's declared API **1.3** with compatible hosts; provider registration methods require a parseable host API in major 1 with minor >=3, without enforcing exact equality. |
| Plugin API | **1.3** | Claim sites, receivers, keyword literals, type vocabulary, and crate deps are API 1.3 contracts. |
| PyTorch | **`torch==2.11.0`** | Same major/minor/patch as the libtorch that published `tch` 0.24.0 expects. |
| Rust crate | **`tch =0.24.0`** with feature **`python-extension`** | Emitted by `crate_dependencies()`; `python-extension` supplies `pyobject_unpack` / `pyobject_wrap` for zero-storage-copy boundaries. |
| Generated Rust crate | Edition **2021**, `rust-version = "1.83"`, PyO3 **0.29** | Inherited from Rextio 0.1.3's generated Cargo manifest. This is an MSRV/API contract, not an exact rustc patch pin. |
| Certified Rust toolchain | `rustc 1.93.1`, `cargo 1.93.1` on `aarch64-apple-darwin` | The real-Cargo Alpha evidence was reproduced with this local toolchain. This repo has no `rust-toolchain.toml`. |
| libtorch linkage | **`LIBTORCH_USE_PYTORCH=1`** | Builds against the active Python torch install. |
| Version-check bypass | **Forbidden** | `LIBTORCH_BYPASS_VERSION_CHECK` is **not** an accepted build path (stripped in e2e env setup; never set). |
| Device | **CPU only** | Boundary extract rejects non-CPU tensors at runtime. |
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
4. **Rextio 0.1.3+ / provider API 1.3** — claim metadata (receivers, literal
   keywords, type keys) is not available on older plugin APIs. The provider
   remains API 1.3 while Core's loader handles compatible host APIs.
   Registration fails closed for API 1.2, other majors, or malformed host
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
# CPython 3.11 venv with the package + pins (torch==2.11.0, rextio>=0.1.3,<0.2)
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'

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

---

## Exactly supported Python forms and result ranks

Claims require API 1.3 metadata that proves the form (operand / receiver type
keys, call target, static keyword literals). Result ranks are those of the
claimed `result_type`.

| Form | Syntax contract | Operand / receiver ranks | Result rank | Rule id (native) |
| --- | --- | --- | --- | --- |
| Functional linear | `torch.nn.functional.linear(x, w, b)` — **three positional** tensors; **no keywords**; **no method form** | x, w: **2**; b: **1** | **2** | `rextio-torch/functional-linear-f32-cpu-2d` |
| ReLU method | `.relu()` — zero args, no keywords | receiver **1** or **2** | **same as receiver** | `…/tensor-relu-f32-cpu-1d` or `…/tensor-relu-f32-cpu-2d` |
| Sigmoid method | `.sigmoid()` — zero args, no keywords | receiver **1** or **2** | **same as receiver** | `…/tensor-sigmoid-f32-cpu-1d` or `…/tensor-sigmoid-f32-cpu-2d` |
| Tanh method | `.tanh()` — zero args, no keywords | receiver **1** or **2** | **same as receiver** | `…/tensor-tanh-f32-cpu-1d` or `…/tensor-tanh-f32-cpu-2d` |
| Matmul binop | `a @ b` | both **2** | **2** | `rextio-torch/tensor-matmul-f32-cpu-2d` |
| Matmul call | `torch.matmul(a, b)` — two positional; no keywords | both **2** | **2** | `rextio-torch/tensor-matmul-call-f32-cpu-2d` |
| Matmul method | `a.matmul(b)` — one positional; no keywords | receiver **2**, other **2** | **2** | same as call form |
| Elementwise `+` (same rank) | `a + b` | both **1**, or both **2** | **same as left** | `rextio-torch/tensor-add-f32-cpu-same-rank` |
| Elementwise `+` (bias broadcast) | `a + b` | one **2**, one **1** (either order) | **2** | `rextio-torch/tensor-add-f32-cpu-2d-1d-broadcast` |
| Mean | `.mean(dim=1, keepdim=False)` — **only** those two **literal** keywords; no positionals | receiver **2** | **1** | `rextio-torch/tensor-mean-dim1-f32-cpu-2d` |
| Sum | `.sum(dim=1, keepdim=False)` — same literal guards as mean | receiver **2** | **1** | `rextio-torch/tensor-sum-dim1-f32-cpu-2d` |

Native op helpers (all fallible, all under `no_grad`):

| Python form | Rust helper | tch API used |
| --- | --- | --- |
| linear | `__rxttorch_linear` | `f_linear` |
| `.relu()` | `__rxttorch_relu` | `f_relu` |
| `.sigmoid()` | `__rxttorch_sigmoid` | `f_sigmoid` |
| `.tanh()` | `__rxttorch_tanh` | `f_tanh` |
| `.mean(…)` | `__rxttorch_mean_dim1_keepdim_false` | `f_mean_dim(1, false, None)` |
| `.sum(…)` | `__rxttorch_sum_dim1_keepdim_false` | `f_sum_dim_intlist(1, false, None)` |
| `+` | `__rxttorch_add` | `f_add` |
| matmul / `@` | `__rxttorch_matmul` | `f_matmul` |

Coverage symbols declared for the analyzer include
`torch.nn.functional.linear`, `torch.matmul`, and method forms
`torch.Tensor.{relu,sigmoid,tanh,mean,sum,matmul}`. Binary `+` / `@` are claimed
via binop sites (not module symbols alone).

### Control flow around claimed ops

Python `for` / `if` with Rextio-lowerable scalar `int` / `bool` conditions become
Rust control flow around the native tch helpers (certified control-flow
vertical slice). Tensor comparisons and tensor-data-dependent conditions are
**not** claimable by plugin API 1.3 and remain on the Python fallback.

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
| Linear: target `torch.nn.functional.linear`, no receiver, exactly 3 positionals, no keywords, ranks (2, 2, 1) | `claim/linear.py` |
| Activations: method form only (receiver present), zero args/keywords, rank 1 or 2 | `claim/activations.py` — module-style `torch.relu` etc. → `NotCovered` |
| Reductions: method form, **no** positionals, keywords exactly `{dim, keepdim}` with **literal** `dim=1` and `keepdim=False`, receiver rank 2 | `claim/reductions.py` |
| Matmul `@` / `torch.matmul` / `.matmul`: both sides rank 2; call forms disallow keywords; method form one positional | `claim/binops.py` |
| Add: binary `+` only; same-rank 1/1 or 2/2, or {1,2} broadcast; other rank pairs rejected | `claim/binops.py` |
| Claim metadata is pure function of site kind, target, operand types, receiver, static keyword literals | `claim/__init__.py` (config unused) |

Keyword order for `dim` / `keepdim` does not matter; values must still be static
literals. Dynamic dim/keepdim, wrong dim, or `keepdim=True` → **Rejected**.

Lowering **independently revalidates** the same metadata (`rule_id`, ranks,
operand counts, receiver) and raises `ValueError` on drift — guards use
exceptions, not `assert`, so they survive `python -O`.

---

## Unsupported / fallback forms

These stay **unclaimed**, **rejected**, or **out of scope**. They must never
become silent native claims.

| Category | Examples / notes | Claim outcome (when offered as a torch site) |
| --- | --- | --- |
| Other devices | CUDA, MPS, non-CPU | Outside vocabulary; boundary would reject non-CPU if a native path were reached |
| Other dtypes | float64, int, etc. | Outside vocabulary; boundary rejects non-float32 at runtime on native paths |
| Other ranks | rank-0 / rank-3+, matmul with rank-1, linear with wrong ranks | `Rejected` when form is recognized with wrong ranks |
| Training / autograd | backward, optimizers, parameter mutation | Out of scope; native helpers always `no_grad` |
| Modules | arbitrary `nn.Module`, module-style activations (`torch.relu`, …) | Uncovered / not claimed |
| Linear variants | keywords, optional bias omission, method linear | `Rejected` or not the linear lane |
| In-place ops | `relu_`, `sigmoid_`, `tanh_`, in-place operators | Not claimed (zero-arg out-of-place methods only; helpers use non-`_` APIs) |
| Elementwise other ops | `-`, `*`, `/`, scalar operands | Not claimed |
| Reductions other shapes | whole-tensor mean/sum, `dim≠1`, `keepdim=True`, dynamic dim/keepdim, positionals | `Rejected` or unclaimed |
| Views / reshape | transpose, view, reshape (alias / shallow-clone risk) | Intentionally not claimed |
| Unsupported broadcast ranks | `+` rank combinations other than same-rank or 2d+1d | `Rejected` |
| Unrelated torch APIs | e.g. `torch.softmax` | `NotCovered` |
| Unresolved types | missing annotation / `None` operand types | `NotCovered` (no false claim) |
| Tensor-dependent control flow | tensor comparisons as `if` conditions | Not claimable under API 1.3 |
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
| Add / broadcast concrete sizes | Fallible `f_add` | same mapping |
| Other op failures | Fallible `f_*` APIs under `no_grad` | same mapping; no `unwrap` / `panic!` / `assert` |

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
activation chaining, and the control-flow route). The claim table also accepts
the reverse **1D+2D** broadcast order; that exact order is unit-tested at the
claim/lower layer but is not a separate real-Cargo fixture.

### Rejected or not covered (illustrative)

```python
# Rejected: linear wrong ranks (1d input)
# F.linear(x_1d, weight_2d, bias_1d)

# Rejected: mean dim / keepdim not the literal contract
# t.mean(dim=0, keepdim=False)
# t.mean(dim=1, keepdim=True)

# Rejected: rank-1 matmul
# a_1d @ b_2d

# NotCovered: unrelated torch API
# torch.softmax(x, dim=-1)

# NotCovered: module-style activation (no receiver on the claim site)
# torch.relu(x)

# Not offered by core to plugins: bare BinOp receiver
# (a @ b + bias).relu()

# Not claimed: in-place / other elementwise / views
# x.relu_();  x * y;  x.transpose(0, 1)
```

---

## Install

Install the exact public Alpha in a CPython 3.11 environment. A source checkout
remains useful for development and for running the focused contract tests.

```bash
# CPython 3.11 only; requires rextio 0.1.3+ and torch 2.11.0
python3.11 -m pip install 'rextio-torch>=0.1.0,<0.2'

# Development checkout alternative
python3.11 -m pip install -e '.[dev]'
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
- Unreleased package metadata identifies version **0.1.1** as Development Status Alpha.

For the longer product definition and phase history, see the
[0.1.0 implementation plan](docs/implementation-plan-0.1.0.md). Historical
benchmark protocols are indexed in [benchmarks/README.md](benchmarks/README.md);
this README is the current support surface.
