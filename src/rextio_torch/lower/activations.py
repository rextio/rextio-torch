"""Lower ``.relu()`` claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.activations import RELU_RULE
from rextio_torch.diagnostics import TENSOR_F32_CPU_2D
from rextio_torch.rust_snippets import RELU, boundary_helpers, relu_helper


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed ReLU method site, or return None."""
    if claimed.kind != "call" or _method_name(claimed.target) != "relu":
        return None
    if claimed.rule_id != RELU_RULE:
        raise ValueError(
            "rextio-torch relu lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {RELU_RULE!r}"
        )
    receiver = claimed.receiver
    if (
        receiver is None
        or receiver.arg_type != TENSOR_F32_CPU_2D
        or claimed.operand_types
        or claimed.keywords
        or claimed.result_type != TENSOR_F32_CPU_2D
    ):
        raise ValueError("rextio-torch received malformed ReLU lower metadata")
    if ctx.receiver is None:
        raise ValueError("rextio-torch relu lower requires ctx.receiver")
    return LoweredExpr(
        rust=f"{RELU}(&{ctx.receiver})?",
        helpers=(boundary_helpers(), relu_helper()),
    )


__all__ = ["try_lower"]
