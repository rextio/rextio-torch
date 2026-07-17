"""Lower activation method claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.activations import (
    ACTIVATION_RULES,
    RELU_RULE,
    RELU_RULE_1D,
    SIGMOID_RULE_1D,
    SIGMOID_RULE_2D,
    TANH_RULE_1D,
    TANH_RULE_2D,
)
from rextio_torch.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_torch.rust_snippets import (
    RELU,
    SIGMOID,
    TANH,
    boundary_helpers,
    relu_helper,
    sigmoid_helper,
    tanh_helper,
)

_METHOD_BY_RULE: dict[str, str] = {
    RELU_RULE: "relu",
    RELU_RULE_1D: "relu",
    SIGMOID_RULE_1D: "sigmoid",
    SIGMOID_RULE_2D: "sigmoid",
    TANH_RULE_1D: "tanh",
    TANH_RULE_2D: "tanh",
}

_HELPER_BY_METHOD: dict[str, tuple[str, str]] = {
    "relu": (RELU, relu_helper()),
    "sigmoid": (SIGMOID, sigmoid_helper()),
    "tanh": (TANH, tanh_helper()),
}

_RANK_TYPES = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed activation method site, or return None."""
    method = _method_name(claimed.target)
    if claimed.kind != "call" or method not in _HELPER_BY_METHOD:
        return None
    if claimed.rule_id not in ACTIVATION_RULES:
        raise ValueError(
            "rextio-torch activation lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    expected_method = _METHOD_BY_RULE.get(claimed.rule_id or "")
    if expected_method != method:
        raise ValueError(
            "rextio-torch activation lower rule/method mismatch: "
            f"rule_id={claimed.rule_id!r} method={method!r}"
        )
    receiver = claimed.receiver
    if (
        receiver is None
        or receiver.arg_type not in _RANK_TYPES
        or claimed.operand_types
        or claimed.keywords
        or claimed.result_type != receiver.arg_type
    ):
        raise ValueError(
            f"rextio-torch received malformed {method} lower metadata"
        )
    if ctx.receiver is None:
        raise ValueError(f"rextio-torch {method} lower requires ctx.receiver")
    call_name, helper = _HELPER_BY_METHOD[method]
    return LoweredExpr(
        rust=f"{call_name}(&{ctx.receiver})?",
        helpers=(boundary_helpers(), helper),
    )


__all__ = ["try_lower"]
