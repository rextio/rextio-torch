"""Import-free public annotation vocabulary for rextio-torch.

These marker classes intentionally import neither torch nor Rextio. Runtime
values remain ordinary ``torch.Tensor`` objects; the integrated Rextio analyzer
resolves the exact dotted annotation spellings to plugin types without executing
the annotations.
"""

from __future__ import annotations

from typing import Any


class TensorF32Cpu2D:
    """A float32 CPU rank-2 tensor (plugin type ``rextio-torch/tensor-f32-cpu-2d``)."""


class TensorF32Cpu1D:
    """A float32 CPU rank-1 tensor (plugin type ``rextio-torch/tensor-f32-cpu-1d``)."""


class TensorI64Cpu1D:
    """An int64 CPU rank-1 tensor (classification-head result only)."""


class TensorF32Cuda0_2D:
    """A float32 CUDA device-0 rank-2 tensor (build-only Alpha contract)."""


class TensorF32Cuda0_1D:
    """A float32 CUDA device-0 rank-1 tensor (build-only Alpha contract)."""


def __getattr__(name: str) -> Any:
    """Reject misspelled annotation names with the normal module error."""
    raise AttributeError(name)


__all__ = [
    "TensorF32Cpu1D",
    "TensorF32Cpu2D",
    "TensorF32Cuda0_1D",
    "TensorF32Cuda0_2D",
    "TensorI64Cpu1D",
]
