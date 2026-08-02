# rextio-torch

<p align="center"><img src="./assets/readme/rextio-icon.png" width="112" alt="Rextio 프로젝트 아이콘"></p>
<p align="center"><strong>범위가 명확한 PyTorch 추론 코드를 Python에서 Rust 기반 <code>tch</code> 연산으로 내립니다.</strong></p>
<p align="center"><a href="https://pypi.org/project/rextio-torch/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-torch?label=PyPI" alt="PyPI의 rextio-torch"></a> <a href="https://github.com/rextio/rextio-torch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT 라이선스"></a></p>
<p align="center"><a href="./README.md">English</a> · <strong>한국어</strong> · <a href="./README.zh-hans.md">简体中文</a> · <a href="./README.zh-hant.md">繁體中文</a> · <a href="./README.ja.md">日本語</a></p>

`rextio-torch`는 [Rextio](https://github.com/rextio/rextio)의 공개 Alpha 플러그인입니다. 의도적으로 좁고 검증된 PyTorch 추론 표면만 인식하여 [`tch`](https://github.com/LaurentMazare/tch-rs) 기반의 실패 가능한 Rust 표현식을 생성합니다. 지원하지 않는 코드는 일반 Python fallback에 남거나 진단과 함께 거부되며 네이티브로 잘못 주장되지 않습니다.

> [!IMPORTANT]
> 추론 전용 CPU 우선 Alpha입니다. 학습, autograd, optimizer, 임의의 `nn.Module` 실행, in-place 텐서 연산은 지원하지 않습니다. CUDA 경로는 build-only이며 `support_claim=false`, `certification_ready=false`입니다.

## 제한된 경로 확인

```python
import torch.nn.functional as F
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D

def inference(x: TensorF32Cpu2D, weight: TensorF32Cpu2D, bias: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
```

Rextio는 `linear → relu → mean`의 rank/type 경로를 증명하고 실패 가능한 `tch` helper로 내립니다. 구체적인 행렬과 broadcast 호환성은 실행 시 libtorch가 검증합니다. 성능 보장은 없습니다. 보존된 Phase A/B 결과는 워크로드 한정의 과거 증거이고 Phase B는 **NO-GO**였으며 0.1.3 small-batch harness는 진단 전용입니다.

## 동작 방식

```text
typed Python function → Rextio 분석/claim → fallible tch helper → PyO3 확장 → PyTorch/libtorch 2.11.0
```

- import-free marker type은 dtype/device/rank만 나타내며 구체적인 차원은 증명하지 않습니다.
- `tch` python-extension bridge의 참조 계수 handle을 사용하며 tensor storage를 복사하지 않습니다.
- Core API 1.7의 적격 PyO3 함수는 입력 변환 후 함수 범위 `tch::no_grad_guard()` 하나를 설치하고 출력 변환 전에 해제합니다.
- RXT075 Python boundary, standalone/type-only, legacy/no-hook 경로는 연산별 guard를 유지합니다.
- 네이티브 출력은 `requires_grad is False`이며 RAII가 성공/오류 시 이전 grad mode를 복원합니다.

## 설치와 첫 사용

```bash
python3.11 -m pip install 'rextio>=0.1.7,<0.2' 'rextio-torch==0.1.3'
export LIBTORCH_USE_PYTORCH=1
unset LIBTORCH_BYPASS_VERSION_CHECK
```

패키지는 `rextio.plugins` entry point로 등록됩니다. 위 예제를 Rextio 프로젝트에서 일반 분석/build 흐름으로 사용하세요. 검색과 `rextio_torch.types`는 PyTorch를 import하지 않지만 네이티브 build에는 일치하는 PyTorch/libtorch와 Rust toolchain이 필요합니다. `LIBTORCH_BYPASS_VERSION_CHECK`는 금지입니다. `PATH`/`VIRTUAL_ENV`의 Python은 PyO3와 같은 CPython 3.11 환경이고 `torch==2.11.0`이어야 합니다.

## 호환성 계약

| 구성 요소 | 계약 |
| --- | --- |
| 패키지 | `rextio-torch==0.1.3` (공개 Alpha, 2026-07-27) |
| CPython | `>=3.11,<3.12` |
| Rextio / API | `>=0.1.7,<0.2` / plugin API `1.7` |
| PyTorch / binding | `torch==2.11.0` / `tch =0.24.0`, `python-extension` |
| 생성 crate | edition 2021, `rust-version = "1.83"`, PyO3 0.29 |
| 인증 toolchain 증거 | `rustc 1.93.1`, `cargo 1.93.1`, `aarch64-apple-darwin` |
| 링크 | `LIBTORCH_USE_PYTORCH=1`; bypass 금지 |
| CPU 텐서 | float32 rank 1/2; 분류 결과 int64 rank 1 가능 |
| 모드 | inference/no-grad 전용 |

macOS arm64는 **Certified Alpha**입니다. Linux x86_64/AArch64는 **Experimental, runtime-backed**이고 비인증입니다. macOS x86_64는 해당 wheel 부재로 availability-gated/unsupported, Linux/macOS i686·ARMv7은 unsupported, Windows는 deferred/unverified입니다.

## 지원되는 CPU 표면

별도 표기가 없으면 operand는 등록된 float32 CPU rank-1/2이고 option은 아래 정적 literal이어야 합니다.

| 계열 | 허용 형식과 경계 |
| --- | --- |
| Linear | `F.linear(x,w,b)` rank 2/2/1; bias 생략, positional `None`, literal `bias=None` |
| Activation | `.relu/.sigmoid/.tanh()`; `torch.relu/sigmoid/tanh`; `F.relu`는 생략/literal `inplace=False`; rank 1/2 |
| Unary | `torch.abs/neg/negative/square/exp/log/sqrt` 또는 대응 zero-arg method; rank 1/2 |
| GELU | `F.gelu(t)`에서 생략 또는 literal `approximate="none"`만 |
| Matmul | `@`, `torch.matmul`, `.matmul`의 2×2, 2×1, 1×2; 1×1 제외 |
| Elementwise | `+ * - /`, `torch.add/sub/mul/div`; 1/1, 2/2, 2/1, 1/2; tensor-tensor, 의미 변경 option 없음 |
| Reduction | method/`torch.mean/sum`; literal `dim=0|1`; `keepdim` 생략/false 또는 named bool; 등록된 rank-1/2 결과만 |
| Softmax | method/`torch.softmax`; `F.softmax`는 생략/literal `dtype=None`; rank-1 dim 0, rank-2 dim 0/1 |
| Argmax | rank-2 dim 0/1 + `keepdim=False`, 또는 rank-1 dim 0 + `keepdim=True`; int64 rank-1 |
| 제어 흐름 | Rextio가 scalar `int`/`bool` 조건을 증명하는 `for`/`if` |

구체적인 차원 오류는 fallible libtorch API를 통해 Python 예외가 됩니다. `log`/`sqrt`의 NaN/infinity/signed-zero 동작은 고정된 eager backend를 따릅니다. Marker는 `TensorF32Cpu2D`, `TensorF32Cpu1D`, `TensorI64Cpu1D`이며 build-only CUDA marker는 `TensorF32Cuda0_2D`, `TensorF32Cuda0_1D`입니다.

## Fallback 또는 fail-closed 대상

다른 dtype/rank/device, transfer, 임의 module, mutation/in-place, view/reshape/transpose, elementwise scalar, tensor-dependent branch, 동적/중복 dim, 의미 변경 option, 등록되지 않은 출력 rank, 무관한 alias는 claim하지 않습니다. 잘못된 정적 shape/option은 `RXTP-TORCH-*`로 거부되어 Python fallback에 남고 무관/미해결 형식은 `NotCovered`입니다. Lowering은 metadata를 다시 검증하고 drift 시 `ValueError`를 냅니다. 네이티브 경계는 잘못된 type/device/dtype/rank/layout을 거부하고 `tch` 오류는 panic이나 조용한 재실행 대신 Python 예외가 됩니다.

## CUDA: 지원이 아닌 증거

후보는 Linux x86_64, CPython 3.11, PyTorch/libtorch 2.11.0, `tch` 0.24.0, 이미 `cuda:0`에 있는 float32 rank-1/2와 다음 경로로 고정됩니다.

```text
rank2 @ rank2 → rank2 + rank1 bias → rank2.relu() → rank2.mean(dim=1) → rank1
```

`rextio-device-cuda/cuda-libtorch-linux-x86_64` authorization이 필요합니다. Hosted CI는 synthetic probe로 compile하지만 extension을 load/execute하지 않습니다. WSL2/RTX 3060 (`sm_86`) 수동 verifier-success evidence는 kernel activity를 포함하지만 여전히 `support_claim=false`, `certification_ready=false`입니다. `.cuda()`/`.to()`, transfer, 혼합 device, multi-GPU, training/autograd, Windows/macOS CUDA, 성능 주장은 없습니다. [고정 CUDA 계약](docs/cuda-build-only-0.1.2.md)을 읽으세요.

## 추가 문서

- [함수 범위 no-grad 계약](docs/invocation-scope-proposal-0.1.3.md)
- [진단 small-batch protocol](docs/preregister-small-batch-scoring-diagnostic-0.1.3.md)
- [과거 benchmark](benchmarks/README.md)
- [변경 기록](CHANGELOG.md)

## 라이선스

[MIT](LICENSE)
