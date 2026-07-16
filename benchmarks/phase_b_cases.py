"""Frozen Phase B deep-control benchmark cases (preregistered 2026-07-17).

The six float32 CPU cells exercise a weight-tied alternating MLP with runtime
``for``/``if`` Python control flow.  Do not add, remove, or tune cells after
observing performance measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Role = Literal["target", "context"]

KERNEL_SOURCE = """\
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch.nn.functional as F


def inference(
    x: TensorF32Cpu2D,
    weight_a: TensorF32Cpu2D,
    weight_b: TensorF32Cpu2D,
    bias_a: TensorF32Cpu1D,
    bias_b: TensorF32Cpu1D,
    depth: int,
    phase: int,
) -> TensorF32Cpu1D:
    hidden = x
    for layer in range(depth):
        if (layer + phase) % 2 == 0:
            hidden = F.linear(hidden, weight_a, bias_a).relu()
        else:
            hidden = F.linear(hidden, weight_b, bias_b).relu()
    return hidden.mean(dim=1, keepdim=False)
"""

CASE_SEED_BASE = 20260717


@dataclass(frozen=True)
class PhaseBBenchmarkCase:
    """One frozen square-hidden Phase B input cell."""

    case_id: str
    role: Role
    batch: int
    width: int
    depth: int
    phase: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation without live tensors."""

        return asdict(self)

    @property
    def x_shape(self) -> tuple[int, int]:
        """Input activation shape."""

        return (self.batch, self.width)

    @property
    def weight_shape(self) -> tuple[int, int]:
        """Both tied weight shapes."""

        return (self.width, self.width)

    @property
    def bias_shape(self) -> tuple[int]:
        """Both tied bias shapes."""

        return (self.width,)

    @property
    def output_shape(self) -> tuple[int]:
        """Output shape after ``mean(dim=1, keepdim=False)``."""

        return (self.batch,)


def _case(
    case_id: str,
    role: Role,
    batch: int,
    width: int,
    depth: int,
    phase: int,
    seed_offset: int,
) -> PhaseBBenchmarkCase:
    return PhaseBBenchmarkCase(
        case_id=case_id,
        role=role,
        batch=batch,
        width=width,
        depth=depth,
        phase=phase,
        seed=CASE_SEED_BASE + seed_offset,
    )


FROZEN_CASES: tuple[PhaseBBenchmarkCase, ...] = (
    _case("t1", "target", batch=1, width=16, depth=4, phase=0, seed_offset=101),
    _case("t2", "target", batch=1, width=32, depth=8, phase=1, seed_offset=102),
    _case("t3", "target", batch=4, width=32, depth=16, phase=0, seed_offset=103),
    _case("t4", "target", batch=8, width=64, depth=16, phase=1, seed_offset=104),
    _case("c1", "context", batch=32, width=128, depth=8, phase=0, seed_offset=105),
    _case("c2", "context", batch=64, width=128, depth=16, phase=1, seed_offset=106),
)

TARGET_CASE_IDS = tuple(case.case_id for case in FROZEN_CASES if case.role == "target")
CONTEXT_CASE_IDS = tuple(case.case_id for case in FROZEN_CASES if case.role == "context")

assert TARGET_CASE_IDS == ("t1", "t2", "t3", "t4")
assert CONTEXT_CASE_IDS == ("c1", "c2")
assert len(FROZEN_CASES) == 6


def frozen_cases() -> list[PhaseBBenchmarkCase]:
    """Return the exact six preregistered cells in report order."""

    return list(FROZEN_CASES)


def get_case(case_id: str) -> PhaseBBenchmarkCase:
    """Return one frozen case or fail on an unknown identifier."""

    for case in FROZEN_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f"unknown frozen Phase B benchmark case: {case_id!r}")


__all__ = [
    "CASE_SEED_BASE",
    "CONTEXT_CASE_IDS",
    "FROZEN_CASES",
    "KERNEL_SOURCE",
    "TARGET_CASE_IDS",
    "PhaseBBenchmarkCase",
    "frozen_cases",
    "get_case",
]
