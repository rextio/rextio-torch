# rextio-torch

<p align="center"><img src="./assets/readme/rextio-icon.png" width="112" alt="Rextio 專案圖示"></p>
<p align="center"><strong>把邊界明確的 PyTorch 推論從 Python lowering 為 Rust 支援的 <code>tch</code> 運算。</strong></p>
<p align="center"><a href="https://pypi.org/project/rextio-torch/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-torch?label=PyPI" alt="PyPI 上的 rextio-torch"></a> <a href="https://github.com/rextio/rextio-torch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT 授權條款"></a></p>
<p align="center"><a href="./README.md">English</a> · <a href="./README.ko.md">한국어</a> · <a href="./README.zh-hans.md">简体中文</a> · <strong>繁體中文</strong> · <a href="./README.ja.md">日本語</a></p>

`rextio-torch` 是 [Rextio](https://github.com/rextio/rextio) 的公開 Alpha 外掛。它只辨識刻意收窄且已驗證的 PyTorch 推論表面，產生由 [`tch`](https://github.com/LaurentMazare/tch-rs) 支援、可回傳錯誤的 Rust 表達式。不支援的程式碼會留在一般 Python fallback，或附帶診斷拒絕，絕不誤報為 native。

> [!IMPORTANT]
> 這是 inference-only、CPU-first Alpha。不支援 training、autograd、optimizer、任意 `nn.Module` 或 in-place 運算。CUDA 僅 build-only，維持 `support_claim=false`、`certification_ready=false`。

## 查看限定路徑

```python
import torch.nn.functional as F
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D

def inference(x: TensorF32Cpu2D, weight: TensorF32Cpu2D, bias: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
```

Rextio 可證明 `linear → relu → mean` 的 rank/type 路徑並 lowering 為 fallible `tch` helper。具體矩陣與 broadcast 相容性由 libtorch 在執行時驗證。沒有效能承諾；Phase A/B 是歷史證據，Phase B 為 **NO-GO**，0.1.3 small-batch harness 僅供診斷。

## 運作方式

```text
typed Python function → Rextio analysis/claim → fallible tch helper → PyO3 extension → PyTorch/libtorch 2.11.0
```

- 無 import marker 只描述 dtype/device/rank，不證明具體維度。
- `tch` python-extension bridge 使用 reference-counted handle，不複製 tensor storage。
- Core API 1.7 的合格 PyO3 函式安裝一個函式範圍 `tch::no_grad_guard()`；RXT075 Python boundary、standalone/type-only、legacy/no-hook 保留逐運算 guard。
- native 輸出 `requires_grad is False`；RAII 在成功或錯誤退出時還原 grad mode。

## 安裝與首次使用

```bash
python3.11 -m pip install 'rextio>=0.1.7,<0.2' 'rextio-torch==0.1.3'
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK
```

套件透過 `rextio.plugins` entry point 註冊。把上例放入 Rextio 專案並使用一般 analysis/build 流程。探索與 `rextio_torch.types` 不匯入 PyTorch，但 native build 需要相符的 PyTorch/libtorch 與 Rust toolchain。禁止 `LIBTORCH_BYPASS_VERSION_CHECK`；`PATH`/`VIRTUAL_ENV` 必須是 PyO3 使用的同一 CPython 3.11 環境，並安裝 `torch==2.11.0`。

## 相容性契約

| 元件 | 契約 |
| --- | --- |
| 套件 | `rextio-torch==0.1.3`（公開 Alpha，2026-07-27） |
| CPython | `>=3.11,<3.12` |
| Rextio / API | `>=0.1.7,<0.2` / plugin API `1.7` |
| PyTorch / binding | `torch==2.11.0` / `tch =0.24.0`, `python-extension` |
| 產生 crate | edition 2021, `rust-version = "1.83"`, PyO3 0.29 |
| 認證 toolchain 證據 | `rustc 1.93.1`, `cargo 1.93.1`, `aarch64-apple-darwin` |
| 連結 | `LIBTORCH_USE_PYTORCH=1`；禁止 bypass |
| CPU tensor / 模式 | float32 rank 1/2；分類結果 int64 rank 1；inference/no-grad |

macOS arm64 是 **Certified Alpha**。Linux x86_64/AArch64 是 **Experimental, runtime-backed**，未認證。macOS x86_64 因缺少 wheel 而 availability-gated/unsupported；i686、ARMv7 不支援；Windows deferred/unverified。

## 支援的 CPU 表面

| 家族 | 接受形式與邊界 |
| --- | --- |
| Linear | `F.linear(x,w,b)` rank 2/2/1；bias 省略、positional `None`、literal `bias=None` |
| Activation | `.relu/.sigmoid/.tanh()`、`torch.relu/sigmoid/tanh`；`F.relu` 僅省略/literal `inplace=False`; rank 1/2 |
| Unary / GELU | `torch.abs/neg/negative/square/exp/log/sqrt` 與對應 zero-arg method；`F.gelu` 僅 `approximate="none"` |
| Matmul | `@`、`torch.matmul`、`.matmul` 的 2×2、2×1、1×2；排除 1×1 |
| Elementwise | `+ * - /`、`torch.add/sub/mul/div`；1/1、2/2、2/1、1/2；tensor-tensor，無改變語意的 option |
| Reduction | method/`torch.mean/sum`；literal `dim=0|1`；`keepdim` 省略/false 或 named bool；已註冊 rank-1/2 結果 |
| Softmax / Argmax | softmax：rank-1 dim 0、rank-2 dim 0/1（`F.softmax` 可省略/literal `dtype=None`）；argmax：rank-2 + `keepdim=False` 或 rank-1 + `keepdim=True`，int64 rank-1 |
| 控制流程 | Rextio 可證明 scalar `int`/`bool` 條件的 `for`/`if` |

具體維度錯誤成為 Python 例外；NaN/infinity/signed-zero 跟隨固定 eager backend。Marker 為 `TensorF32Cpu2D`、`TensorF32Cpu1D`、`TensorI64Cpu1D`；build-only CUDA marker 為 `TensorF32Cuda0_2D`、`TensorF32Cuda0_1D`。

## Fallback 與 fail-closed

不 claim 其他 dtype/rank/device、transfer、任意 module、mutation/in-place、view/reshape/transpose、elementwise scalar、tensor-dependent branch、動態/重複 dim、改變語意的 option、未註冊輸出 rank 或無關 alias。錯誤靜態 shape/option 用 `RXTP-TORCH-*` 拒絕並留在 Python fallback；無關或未解析為 `NotCovered`。Lowering 重驗 metadata，漂移時 `ValueError`；native boundary 拒絕錯誤 type/device/dtype/rank/layout，`tch` 錯誤映射為 Python 例外。

## CUDA：證據，不是支援

僅限 Linux x86_64、CPython 3.11、PyTorch/libtorch 2.11.0、`tch` 0.24.0、已駐留 `cuda:0` 的 float32 rank-1/2 與：

```text
rank2 @ rank2 → rank2 + rank1 bias → rank2.relu() → rank2.mean(dim=1) → rank1
```

需要 `rextio-device-cuda/cuda-libtorch-linux-x86_64` authorization。Hosted CI 用 synthetic probe 編譯但不 load/execute。WSL2/RTX 3060 (`sm_86`) 的手動 verifier-success evidence 包含 kernel activity，仍是 `support_claim=false`、`certification_ready=false`。不含 `.cuda()`/`.to()`、transfer、混合 device、multi-GPU、training/autograd、Windows/macOS CUDA 或效能聲明。見[固定 CUDA 契約](docs/cuda-build-only-0.1.2.md)。

## 更多文件

- [函式範圍 no-grad 契約](docs/invocation-scope-proposal-0.1.3.md)
- [診斷 small-batch protocol](docs/preregister-small-batch-scoring-diagnostic-0.1.3.md)
- [歷史 benchmark](benchmarks/README.md)
- [變更記錄](CHANGELOG.md)

## 授權

[MIT](LICENSE)
