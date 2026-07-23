"""Lower activation method claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.activations import (
    ACTIVATION_RULES,
    FUNCTION_RELU_RULE,
    FUNCTION_SIGMOID_RULE,
    FUNCTION_TANH_RULE,
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
    FUNCTION_RELU_RULE: "relu",
    FUNCTION_SIGMOID_RULE: "sigmoid",
    FUNCTION_TANH_RULE: "tanh",
}

_HELPER_BY_METHOD: dict[str, tuple[str, str]] = {
    "relu": (RELU, relu_helper()),
    "sigmoid": (SIGMOID, sigmoid_helper()),
    "tanh": (TANH, tanh_helper()),
}

_RANK_TYPES = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})
_FUNCTION_TARGETS: dict[str, str] = {
    "torch.relu": "relu",
    "torch.sigmoid": "sigmoid",
    "torch.tanh": "tanh",
}
_FUNCTION_RULES = frozenset(
    {FUNCTION_RELU_RULE, FUNCTION_SIGMOID_RULE, FUNCTION_TANH_RULE}
)


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
    if claimed.rule_id in _FUNCTION_RULES:
        if (
            _FUNCTION_TARGETS.get(claimed.target) != method
            or claimed.receiver is not None
            or ctx.receiver is not None
            or len(claimed.operand_types) != 1
            or len(ctx.operands) != 1
            or claimed.operand_types[0] not in _RANK_TYPES
            or claimed.keywords
            or claimed.result_type != claimed.operand_types[0]
        ):
            raise ValueError(
                f"rextio-torch received malformed functional {method} lower metadata"
            )
        input_name = ctx.operands[0]
    else:
        receiver = claimed.receiver
        if (
            receiver is None
            or receiver.arg_type not in _RANK_TYPES
            or claimed.operand_types
            or claimed.keywords
            or claimed.result_type != receiver.arg_type
            or ctx.receiver is None
            or ctx.operands
        ):
            raise ValueError(
                f"rextio-torch received malformed method {method} lower metadata"
            )
        input_name = ctx.receiver
    call_name, helper = _HELPER_BY_METHOD[method]
    return LoweredExpr(
        rust=f"{call_name}(&{input_name})?",
        helpers=(boundary_helpers(), helper),
    )


__all__ = ["try_lower"]
