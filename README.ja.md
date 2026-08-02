# rextio-torch

<p align="center"><img src="./assets/readme/rextio-icon.png" width="112" alt="Rextio プロジェクトアイコン"></p>
<p align="center"><strong>範囲を限定した PyTorch 推論を Python から Rust ベースの <code>tch</code> 演算へ lowering します。</strong></p>
<p align="center"><a href="https://pypi.org/project/rextio-torch/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-torch?label=PyPI" alt="PyPI の rextio-torch"></a> <a href="https://github.com/rextio/rextio-torch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT ライセンス"></a></p>
<p align="center"><a href="./README.md">English</a> · <a href="./README.ko.md">한국어</a> · <a href="./README.zh-hans.md">简体中文</a> · <a href="./README.zh-hant.md">繁體中文</a> · <strong>日本語</strong></p>

`rextio-torch` は [Rextio](https://github.com/rextio/rextio) の公開 Alpha プラグインです。意図的に狭く検証済みの PyTorch 推論面だけを認識し、[`tch`](https://github.com/LaurentMazare/tch-rs) による fallible な Rust 式を生成します。対象外コードは通常の Python fallback に残るか診断付きで拒否され、誤って native と宣言されません。

> [!IMPORTANT]
> 推論専用・CPU 優先です。training、autograd、optimizer、任意の `nn.Module`、in-place 演算は非対応です。CUDA は build-only で `support_claim=false`、`certification_ready=false` です。

## 限定経路を見る

```python
import torch.nn.functional as F
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D

def inference(x: TensorF32Cpu2D, weight: TensorF32Cpu2D, bias: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
```

Rextio は `linear → relu → mean` の rank/type 経路を証明し、fallible な `tch` helper に lowering します。具体的な行列・broadcast は実行時に libtorch が検証します。性能保証はなく、Phase A/B は履歴証拠、Phase B は **NO-GO**、0.1.3 small-batch harness は診断専用です。

## 仕組み

```text
typed Python function → Rextio analysis/claim → fallible tch helper → PyO3 extension → PyTorch/libtorch 2.11.0
```

- import-free marker は dtype/device/rank のみを示します。
- `tch` python-extension bridge の reference-counted handle を使い、storage をコピーしません。
- Core API 1.7 の対象 PyO3 関数は関数スコープ `tch::no_grad_guard()` を 1 個使います。RXT075 Python boundary、standalone/type-only、legacy/no-hook は演算ごとの guard を維持します。
- native 出力は `requires_grad is False` で、RAII が元の grad mode を復元します。

## インストールと最初の利用

```bash
python3.11 -m pip install 'rextio>=0.1.7,<0.2' 'rextio-torch==0.1.3'
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK
```

`rextio.plugins` entry point で登録されます。上例を通常の Rextio analysis/build フローで使います。探索と `rextio_torch.types` は PyTorch を import しませんが、native build には一致する PyTorch/libtorch と Rust toolchain が必要です。`LIBTORCH_BYPASS_VERSION_CHECK` は禁止で、`PATH`/`VIRTUAL_ENV` は PyO3 と同じ CPython 3.11、`torch==2.11.0` でなければなりません。

## 互換性契約

| Component | 契約 |
| --- | --- |
| Package | `rextio-torch==0.1.3`（公開 Alpha、2026-07-27） |
| CPython | `>=3.11,<3.12` |
| Rextio / API | `>=0.1.7,<0.2` / plugin API `1.7` |
| PyTorch / binding | `torch==2.11.0` / `tch =0.24.0`, `python-extension` |
| 生成 crate | edition 2021, `rust-version = "1.83"`, PyO3 0.29 |
| 認証 evidence | `rustc 1.93.1`, `cargo 1.93.1`, `aarch64-apple-darwin` |
| Link | `LIBTORCH_USE_PYTORCH=1`; bypass 禁止 |
| CPU tensor / mode | float32 rank 1/2、分類結果 int64 rank 1；inference/no-grad |

macOS arm64 は **Certified Alpha**。Linux x86_64/AArch64 は **Experimental, runtime-backed** で未認証です。macOS x86_64 は wheel 不在で availability-gated/unsupported、i686/ARMv7 は unsupported、Windows は deferred/unverified です。

## 対応 CPU 面

| 系統 | 対応形と境界 |
| --- | --- |
| Linear | `F.linear(x,w,b)` rank 2/2/1、bias 省略・positional `None`・literal `bias=None` |
| Activation | `.relu/.sigmoid/.tanh()`、`torch.relu/sigmoid/tanh`、`F.relu` は省略/literal `inplace=False`; rank 1/2 |
| Unary / GELU | `torch.abs/neg/negative/square/exp/log/sqrt` と対応 method; `F.gelu` は `approximate="none"` のみ |
| Matmul | `@`, `torch.matmul`, `.matmul` の 2×2, 2×1, 1×2; 1×1 除外 |
| Elementwise | `+ * - /`, `torch.add/sub/mul/div`; 1/1, 2/2, 2/1, 1/2; tensor-tensor のみ |
| Reduction | method/`torch.mean/sum`; literal `dim=0|1`; `keepdim` 省略/false または named bool; 登録済み rank-1/2 のみ |
| Softmax / Argmax | softmax は rank-1 dim 0、rank-2 dim 0/1（`F.softmax` の `dtype=None` 可）。argmax は rank-2 + `keepdim=False` または rank-1 + `keepdim=True`、int64 rank-1 |
| Control flow | scalar `int`/`bool` を Rextio が証明できる `for`/`if` |

具体的次元エラーは Python 例外になり、NaN/infinity/signed-zero は固定 eager backend に従います。Marker は `TensorF32Cpu2D`, `TensorF32Cpu1D`, `TensorI64Cpu1D`、build-only CUDA marker は `TensorF32Cuda0_2D`, `TensorF32Cuda0_1D` です。

## Fallback / fail-closed

他の dtype/rank/device、transfer、任意 module、mutation/in-place、view/reshape/transpose、scalar elementwise、tensor-dependent branch、動的/重複 dim、意味変更 option、未登録 output rank、別 alias は claim しません。不正な静的 shape/option は `RXTP-TORCH-*` で拒否して Python fallback に残し、未解決は `NotCovered`。Lowering は metadata を再検証して drift に `ValueError`、native boundary は type/device/dtype/rank/layout を検査し、`tch` error は Python 例外になります。

## CUDA：support ではなく evidence

Linux x86_64、CPython 3.11、PyTorch/libtorch 2.11.0、`tch` 0.24.0、すでに `cuda:0` 上の float32 rank-1/2 と次だけです。

```text
rank2 @ rank2 → rank2 + rank1 bias → rank2.relu() → rank2.mean(dim=1) → rank1
```

`rextio-device-cuda/cuda-libtorch-linux-x86_64` authorization が必要です。Hosted CI は synthetic probe で compile するだけです。WSL2/RTX 3060 (`sm_86`) の手動 evidence は kernel activity を含みますが `support_claim=false`, `certification_ready=false` のままです。transfer、混在 device、multi-GPU、training/autograd、Windows/macOS CUDA、性能主張はありません。[固定 CUDA 契約](docs/cuda-build-only-0.1.2.md)を参照してください。

## 詳細

- [関数スコープ no-grad 契約](docs/invocation-scope-proposal-0.1.3.md)
- [診断 small-batch protocol](docs/preregister-small-batch-scoring-diagnostic-0.1.3.md)
- [履歴 benchmark](benchmarks/README.md)
- [変更履歴](CHANGELOG.md)

## ライセンス

[MIT](LICENSE)
