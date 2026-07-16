"""Frozen Phase A product-route benchmark cases (preregistered 2026-07-17).

Exactly six float32 CPU cells for the linear→ReLU→mean(dim=1) product route.
Shapes and seeds are fixed before measurement; do not add, remove, or retune
cells after observing timings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Role = Literal["target", "context"]

# Same annotated product surface as the Phase A real-Cargo E2E certification.
KERNEL_SOURCE = """
from rextio_torch.types import TensorF32Cpu1D, TensorF32Cpu2D
import torch.nn.functional as F


def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    return F.linear(x, weight, bias).relu().mean(dim=1, keepdim=False)
"""

# Shared seed base for deterministic x / weight / bias per cell.
CASE_SEED_BASE = 20260717


@dataclass(frozen=True)
class BenchmarkCase:
    """One identical native/fallback input cell with fixed shapes and seed."""

    case_id: str
    role: Role
    batch: int
    in_features: int
    out_features: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation (no live tensors)."""
        return asdict(self)

    @property
    def x_shape(self) -> tuple[int, int]:
        """Input activation shape ``(batch, in_features)``."""
        return (self.batch, self.in_features)

    @property
    def weight_shape(self) -> tuple[int, int]:
        """Linear weight shape ``(out_features, in_features)``."""
        return (self.out_features, self.in_features)

    @property
    def bias_shape(self) -> tuple[int]:
        """Bias shape ``(out_features,)``."""
        return (self.out_features,)

    @property
    def output_shape(self) -> tuple[int]:
        """Expected product output shape after mean(dim=1)."""
        return (self.batch,)


def _case(
    case_id: str,
    role: Role,
    batch: int,
    in_features: int,
    out_features: int,
    *,
    seed_offset: int,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        role=role,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
        seed=CASE_SEED_BASE + seed_offset,
    )


# Frozen matrix — order is authoritative for reporting.
# Target niche (GO gate): t1–t4. Context sizes (primary path still measured): c1–c2.
FROZEN_CASES: tuple[BenchmarkCase, ...] = (
    _case("t1", "target", batch=1, in_features=32, out_features=32, seed_offset=1),
    _case("t2", "target", batch=1, in_features=128, out_features=64, seed_offset=2),
    _case("t3", "target", batch=8, in_features=64, out_features=64, seed_offset=3),
    _case("t4", "target", batch=16, in_features=128, out_features=64, seed_offset=4),
    _case("c1", "context", batch=64, in_features=128, out_features=64, seed_offset=5),
    _case("c2", "context", batch=256, in_features=256, out_features=128, seed_offset=6),
)

TARGET_CASE_IDS: tuple[str, ...] = tuple(
    case.case_id for case in FROZEN_CASES if case.role == "target"
)
CONTEXT_CASE_IDS: tuple[str, ...] = tuple(
    case.case_id for case in FROZEN_CASES if case.role == "context"
)

assert TARGET_CASE_IDS == ("t1", "t2", "t3", "t4")
assert CONTEXT_CASE_IDS == ("c1", "c2")
assert len(FROZEN_CASES) == 6


def frozen_cases() -> list[BenchmarkCase]:
    """Return the exact six preregistered cells in report order."""
    return list(FROZEN_CASES)


def get_case(case_id: str) -> BenchmarkCase:
    """Look up one frozen cell by id; raise on unknown ids."""
    for case in FROZEN_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f"unknown frozen benchmark case: {case_id!r}")


__all__ = [
    "CASE_SEED_BASE",
    "CONTEXT_CASE_IDS",
    "FROZEN_CASES",
    "KERNEL_SOURCE",
    "TARGET_CASE_IDS",
    "BenchmarkCase",
    "frozen_cases",
    "get_case",
]
