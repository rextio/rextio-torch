"""Diagnostic small-batch scoring pipeline cases (0.1.3).

Self-contained product-shaped source: rank-2/rank-1 normalize (sub, div),
functional linear, scalar ``range`` + integer ``if`` routing through ReLU and
tanh, classification linear, softmax, argmax.

``rounds`` is the number of scalar control-flow iterations in the pipeline
(predeclared as 4 for every cell). Timing sample counts are separate and live
in the harness, not in this matrix.

This is **not** an official performance cohort and must never overwrite
Phase A/B historical results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

# Full pipeline inlined three times so each entry point is a standalone
# plugin-typed native candidate (RXT092: native cannot call another
# plugin-typed function).
KERNEL_SOURCE = """\
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D, TensorI64Cpu1D
import torch.nn.functional as F


def score_logits(
    x: TensorF32Cpu2D,
    mean: TensorF32Cpu1D,
    scale: TensorF32Cpu1D,
    hidden_weight: TensorF32Cpu2D,
    hidden_bias: TensorF32Cpu1D,
    class_weight: TensorF32Cpu2D,
    class_bias: TensorF32Cpu1D,
    rounds: int,
    phase: int,
) -> TensorF32Cpu2D:
    centered = x - mean
    normalized = centered / scale
    hidden = F.linear(normalized, hidden_weight, hidden_bias)
    for layer in range(rounds):
        if (layer + phase) % 2 == 0:
            hidden = hidden.relu()
        else:
            hidden = hidden.tanh()
    return F.linear(hidden, class_weight, class_bias)


def score_probabilities(
    x: TensorF32Cpu2D,
    mean: TensorF32Cpu1D,
    scale: TensorF32Cpu1D,
    hidden_weight: TensorF32Cpu2D,
    hidden_bias: TensorF32Cpu1D,
    class_weight: TensorF32Cpu2D,
    class_bias: TensorF32Cpu1D,
    rounds: int,
    phase: int,
) -> TensorF32Cpu2D:
    centered = x - mean
    normalized = centered / scale
    hidden = F.linear(normalized, hidden_weight, hidden_bias)
    for layer in range(rounds):
        if (layer + phase) % 2 == 0:
            hidden = hidden.relu()
        else:
            hidden = hidden.tanh()
    logits = F.linear(hidden, class_weight, class_bias)
    return logits.softmax(dim=1)


def score_labels(
    x: TensorF32Cpu2D,
    mean: TensorF32Cpu1D,
    scale: TensorF32Cpu1D,
    hidden_weight: TensorF32Cpu2D,
    hidden_bias: TensorF32Cpu1D,
    class_weight: TensorF32Cpu2D,
    class_bias: TensorF32Cpu1D,
    rounds: int,
    phase: int,
) -> TensorI64Cpu1D:
    centered = x - mean
    normalized = centered / scale
    hidden = F.linear(normalized, hidden_weight, hidden_bias)
    for layer in range(rounds):
        if (layer + phase) % 2 == 0:
            hidden = hidden.relu()
        else:
            hidden = hidden.tanh()
    logits = F.linear(hidden, class_weight, class_bias)
    probabilities = logits.softmax(dim=1)
    return probabilities.argmax(dim=1, keepdim=False)
"""

FEATURE_WIDTH: Final[int] = 32
CLASS_COUNT: Final[int] = 8
HIDDEN_WIDTH: Final[int] = 32
# Scalar control-flow iterations inside the pipeline (not timing samples).
CONTROL_FLOW_ROUNDS: Final[int] = 4
CONTROL_PHASE: Final[int] = 0
# Default diagnostic timing sample count (independent of control-flow rounds).
DEFAULT_TIMING_SAMPLES: Final[int] = 4
BATCH_SIZES: Final[tuple[int, ...]] = (1, 16, 128)
CASE_SEED_BASE: Final[int] = 20260727

LOGITS_RTOL: Final[float] = 1e-5
LOGITS_ATOL: Final[float] = 1e-6
PROBS_RTOL: Final[float] = 1e-5
PROBS_ATOL: Final[float] = 1e-6

EXPECTED_CLAIM_RULE_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "rextio-torch/tensor-sub-f32-cpu-2d-1d-broadcast",
        "rextio-torch/tensor-div-f32-cpu-2d-1d-broadcast",
        "rextio-torch/functional-linear-f32-cpu-2d",
        "rextio-torch/tensor-relu-f32-cpu-2d",
        "rextio-torch/tensor-tanh-f32-cpu-2d",
        "rextio-torch/tensor-softmax-dim1-f32-cpu-2d",
        "rextio-torch/tensor-argmax-dim1-keepfalse-i64-cpu-1d",
    }
)

# Qualnames used by the real-Cargo fixture package (torch_score).
SCORE_LOGITS_QUALNAME: Final[str] = "torch_score.kernels.score_logits"
SCORE_PROBS_QUALNAME: Final[str] = "torch_score.kernels.score_probabilities"
SCORE_LABELS_QUALNAME: Final[str] = "torch_score.kernels.score_labels"
SCORE_MODULE_NAME: Final[str] = "torch_score.kernels"
EXPECTED_NATIVE_ROUTE: Final[str] = "native-plugin:rextio-torch"


@dataclass(frozen=True)
class SmallBatchScoringCase:
    """One predeclared diagnostic cell (shapes only; no live tensors)."""

    case_id: str
    batch: int
    feature_width: int
    hidden_width: int
    class_count: int
    rounds: int
    phase: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation without live tensors."""
        return asdict(self)

    @property
    def x_shape(self) -> tuple[int, int]:
        """Input activations ``(batch, feature_width)``."""
        return (self.batch, self.feature_width)

    @property
    def mean_shape(self) -> tuple[int]:
        """Rank-1 normalization mean ``(feature_width,)``."""
        return (self.feature_width,)

    @property
    def scale_shape(self) -> tuple[int]:
        """Rank-1 normalization scale ``(feature_width,)``."""
        return (self.feature_width,)

    @property
    def hidden_weight_shape(self) -> tuple[int, int]:
        """Hidden linear weight ``(hidden_width, feature_width)``."""
        return (self.hidden_width, self.feature_width)

    @property
    def hidden_bias_shape(self) -> tuple[int]:
        """Hidden linear bias ``(hidden_width,)``."""
        return (self.hidden_width,)

    @property
    def class_weight_shape(self) -> tuple[int, int]:
        """Classification weight ``(class_count, hidden_width)``."""
        return (self.class_count, self.hidden_width)

    @property
    def class_bias_shape(self) -> tuple[int]:
        """Classification bias ``(class_count,)``."""
        return (self.class_count,)

    @property
    def logits_shape(self) -> tuple[int, int]:
        """Logits ``(batch, class_count)``."""
        return (self.batch, self.class_count)

    @property
    def labels_shape(self) -> tuple[int]:
        """Argmax labels ``(batch,)``."""
        return (self.batch,)


def _case(case_id: str, batch: int, *, seed_offset: int) -> SmallBatchScoringCase:
    return SmallBatchScoringCase(
        case_id=case_id,
        batch=batch,
        feature_width=FEATURE_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        class_count=CLASS_COUNT,
        rounds=CONTROL_FLOW_ROUNDS,
        phase=CONTROL_PHASE,
        seed=CASE_SEED_BASE + seed_offset,
    )


FROZEN_CASES: tuple[SmallBatchScoringCase, ...] = (
    _case("s1", batch=1, seed_offset=1),
    _case("s16", batch=16, seed_offset=2),
    _case("s128", batch=128, seed_offset=3),
)

assert [case.batch for case in FROZEN_CASES] == list(BATCH_SIZES)
assert all(case.feature_width == FEATURE_WIDTH for case in FROZEN_CASES)
assert all(case.class_count == CLASS_COUNT for case in FROZEN_CASES)
assert all(case.rounds == CONTROL_FLOW_ROUNDS == 4 for case in FROZEN_CASES)
assert DEFAULT_TIMING_SAMPLES >= 1


def frozen_cases() -> list[SmallBatchScoringCase]:
    """Return the exact three predeclared diagnostic cells in report order."""
    return list(FROZEN_CASES)


def get_case(case_id: str) -> SmallBatchScoringCase:
    """Look up one diagnostic cell by id; raise on unknown ids."""
    for case in FROZEN_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f"unknown small-batch scoring case: {case_id!r}")


__all__ = [
    "BATCH_SIZES",
    "CASE_SEED_BASE",
    "CLASS_COUNT",
    "CONTROL_FLOW_ROUNDS",
    "CONTROL_PHASE",
    "DEFAULT_TIMING_SAMPLES",
    "EXPECTED_CLAIM_RULE_FAMILIES",
    "EXPECTED_NATIVE_ROUTE",
    "FEATURE_WIDTH",
    "FROZEN_CASES",
    "HIDDEN_WIDTH",
    "KERNEL_SOURCE",
    "LOGITS_ATOL",
    "LOGITS_RTOL",
    "PROBS_ATOL",
    "PROBS_RTOL",
    "SCORE_LABELS_QUALNAME",
    "SCORE_LOGITS_QUALNAME",
    "SCORE_MODULE_NAME",
    "SCORE_PROBS_QUALNAME",
    "SmallBatchScoringCase",
    "frozen_cases",
    "get_case",
]
