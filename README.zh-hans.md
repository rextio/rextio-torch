# rextio-torch

<p align="center"><img src="./assets/readme/rextio-icon.png" width="112" alt="Rextio 项目图标"></p>
<p align="center"><strong>将边界明确的 PyTorch 推理从 Python lowering 为 Rust 支持的 <code>tch</code> 运算。</strong></p>
<p align="center"><a href="https://pypi.org/project/rextio-torch/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-torch?label=PyPI" alt="PyPI 上的 rextio-torch"></a> <a href="https://github.com/rextio/rextio-torch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT 许可证"></a></p>
<p align="center"><a href="./README.md">English</a> · <a href="./README.ko.md">한국어</a> · <strong>简体中文</strong> · <a href="./README.zh-hant.md">繁體中文</a> · <a href="./README.ja.md">日本語</a></p>

`rextio-torch` 是 [Rextio](https://github.com/rextio/rextio) 的公开 Alpha 插件。它只识别刻意收窄且经过验证的 PyTorch 推理表面，并生成由 [`tch`](https://github.com/LaurentMazare/tch-rs) 支持、可返回错误的 Rust 表达式。不支持的代码会留在普通 Python fallback，或带诊断拒绝，绝不会被误报为 native。

> [!IMPORTANT]
> 这是 inference-only、CPU-first Alpha。不支持训练、autograd、optimizer、任意 `nn.Module` 或 in-place 运算。CUDA 仅 build-only，且保持 `support_claim=false`、`certification_ready=false`。

## 查看限定路径

```python
import torch.nn.functional as F
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D

def inference(x: TensorF32Cpu2D, weight: TensorF32Cpu2D, bias: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
```

Rextio 可证明 `linear → relu → mean` 的 rank/type 路径，并 lowering 为 fallible `tch` helper。具体矩阵和 broadcast 兼容性仍由 libtorch 在运行时验证。没有性能承诺；Phase A/B 是工作负载限定的历史证据，Phase B 为 **NO-GO**，0.1.3 small-batch harness 仅用于诊断。

## 工作原理

```text
typed Python function → Rextio analysis/claim → fallible tch helper → PyO3 extension → PyTorch/libtorch 2.11.0
```

- 无 import marker 只描述 dtype/device/rank，不证明具体维度。
- 边界使用 `tch` python-extension bridge 的引用计数 handle，不复制 tensor storage。
- Core API 1.7 的合格 PyO3 函数在输入转换后安装一个函数作用域 `tch::no_grad_guard()`，并在输出转换前释放。
- RXT075 Python boundary、standalone/type-only、legacy/no-hook 路径保留逐运算 guard。
- native 输出 `requires_grad is False`；RAII 在成功或错误退出时恢复原 grad mode。

## 安装与首次使用

```bash
python3.11 -m pip install 'rextio>=0.1.7,<0.2' 'rextio-torch==0.1.3'
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK
```

包通过 `rextio.plugins` entry point 注册。把上例放进 Rextio 项目并使用正常 analysis/build 流程。发现插件和导入 `rextio_torch.types` 不会导入 PyTorch，但 native build 需要匹配的 PyTorch/libtorch 和 Rust toolchain。禁止 `LIBTORCH_BYPASS_VERSION_CHECK`；`PATH`/`VIRTUAL_ENV` 中的 Python 必须是 PyO3 使用的同一 CPython 3.11 环境，并安装 `torch==2.11.0`。

## 兼容性契约

| 组件 | 契约 |
| --- | --- |
| 包 | `rextio-torch==0.1.3`（公开 Alpha，2026-07-27） |
| CPython | `>=3.11,<3.12` |
| Rextio / API | `>=0.1.7,<0.2` / plugin API `1.7` |
| PyTorch / binding | `torch==2.11.0` / `tch =0.24.0`, `python-extension` |
| 生成 crate | edition 2021, `rust-version = "1.83"`, PyO3 0.29 |
| 认证 toolchain 证据 | `rustc 1.93.1`, `cargo 1.93.1`, `aarch64-apple-darwin` |
| 链接 | `LIBTORCH_USE_PYTORCH=1`；禁止 bypass |
| CPU tensor | float32 rank 1/2；分类结果可为 int64 rank 1 |
| 模式 | 仅 inference/no-grad |

macOS arm64 是 **Certified Alpha**。Linux x86_64/AArch64 是 **Experimental, runtime-backed**，未认证。macOS x86_64 因缺少精确 wheel 而 availability-gated/unsupported；Linux/macOS i686、ARMv7 不支持；Windows deferred/unverified。

## 支持的 CPU 表面

除非另注，operand 必须是已注册的 float32 CPU rank-1/2，option 必须是下列静态 literal。

| 家族 | 接受形式与边界 |
| --- | --- |
| Linear | `F.linear(x,w,b)` rank 2/2/1；或省略 bias、positional `None`、literal `bias=None` |
| Activation | `.relu/.sigmoid/.tanh()`；`torch.relu/sigmoid/tanh`；`F.relu` 仅省略/literal `inplace=False`；rank 1/2 |
| Unary | `torch.abs/neg/negative/square/exp/log/sqrt` 或对应 zero-arg method；rank 1/2 |
| GELU | `F.gelu(t)` 仅省略或 literal `approximate="none"` |
| Matmul | `@`、`torch.matmul`、`.matmul` 的 2×2、2×1、1×2；排除 1×1 |
| Elementwise | `+ * - /`、`torch.add/sub/mul/div`；1/1、2/2、2/1、1/2；仅 tensor-tensor，无改变语义的 option |
| Reduction | method/`torch.mean/sum`；literal `dim=0|1`；`keepdim` 省略/false 或 named bool；只允许已注册 rank-1/2 结果 |
| Softmax | method/`torch.softmax`；`F.softmax` 允许省略/literal `dtype=None`；rank-1 dim 0、rank-2 dim 0/1 |
| Argmax | rank-2 dim 0/1 + `keepdim=False`，或 rank-1 dim 0 + `keepdim=True`；int64 rank-1 |
| 控制流 | Rextio 可证明 scalar `int`/`bool` 条件的 `for`/`if` |

具体维度错误通过 fallible libtorch API 成为 Python 异常。`log`/`sqrt` 的 NaN/infinity/signed-zero 行为跟随固定 eager backend。Marker 为 `TensorF32Cpu2D`、`TensorF32Cpu1D`、`TensorI64Cpu1D`；build-only CUDA marker 为 `TensorF32Cuda0_2D`、`TensorF32Cuda0_1D`。

## Fallback 与 fail-closed

插件不 claim 其他 dtype/rank/device、transfer、任意 module、mutation/in-place、view/reshape/transpose、elementwise scalar、tensor-dependent branch、动态/重复 dim、改变语义的 option、未注册输出 rank 或无关 alias。错误静态 shape/option 用 `RXTP-TORCH-*` 拒绝并留在 Python fallback；无关或未解析形式为 `NotCovered`。Lowering 重新验证 metadata，漂移时抛出 `ValueError`。native boundary 拒绝错误 type/device/dtype/rank/layout，`tch` 错误映射为 Python 异常，而非 panic 或静默重放。

## CUDA：证据，不是支持

候选仅限 Linux x86_64、CPython 3.11、PyTorch/libtorch 2.11.0、`tch` 0.24.0、已驻留 `cuda:0` 的 float32 rank-1/2，以及：

```text
rank2 @ rank2 → rank2 + rank1 bias → rank2.relu() → rank2.mean(dim=1) → rank1
```

需要 `rextio-device-cuda/cuda-libtorch-linux-x86_64` authorization。Hosted CI 用 synthetic probe 编译，但不 load/execute extension。保留的 WSL2/RTX 3060 (`sm_86`) 手动 verifier-success evidence 包括 kernel activity，但仍为 `support_claim=false`、`certification_ready=false`。不包含 `.cuda()`/`.to()`、transfer、混合 device、multi-GPU、training/autograd、Windows/macOS CUDA 或性能声明。请先阅读[固定 CUDA 契约](docs/cuda-build-only-0.1.2.md)。

## 更多文档

- [函数作用域 no-grad 契约](docs/invocation-scope-proposal-0.1.3.md)
- [诊断 small-batch protocol](docs/preregister-small-batch-scoring-diagnostic-0.1.3.md)
- [历史 benchmark](benchmarks/README.md)
- [变更记录](CHANGELOG.md)

## 许可证

[MIT](LICENSE)
