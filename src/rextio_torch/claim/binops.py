"""Fail-closed claims for elementwise binary ops and rank-2 matmul."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_ADD,
    DIAGNOSTIC_DIV,
    DIAGNOSTIC_FUNCTION_ADD,
    DIAGNOSTIC_FUNCTION_DIV,
    DIAGNOSTIC_FUNCTION_MUL,
    DIAGNOSTIC_FUNCTION_SUB,
    DIAGNOSTIC_MUL,
    DIAGNOSTIC_MATMUL,
    DIAGNOSTIC_MATMUL_CALL,
    DIAGNOSTIC_MATMUL_CALL_MIXED_RANK,
    DIAGNOSTIC_MATMUL_MIXED_RANK,
    DIAGNOSTIC_SUB,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

ADD_SAME_RANK_RULE = "rextio-torch/tensor-add-f32-cpu-same-rank"
ADD_BROADCAST_2D_1D_RULE = "rextio-torch/tensor-add-f32-cpu-2d-1d-broadcast"
MUL_SAME_RANK_RULE = "rextio-torch/tensor-mul-f32-cpu-same-rank"
MUL_BROADCAST_2D_1D_RULE = "rextio-torch/tensor-mul-f32-cpu-2d-1d-broadcast"
SUB_SAME_RANK_RULE = "rextio-torch/tensor-sub-f32-cpu-same-rank"
SUB_BROADCAST_2D_1D_RULE = "rextio-torch/tensor-sub-f32-cpu-2d-1d-broadcast"
DIV_SAME_RANK_RULE = "rextio-torch/tensor-div-f32-cpu-same-rank"
DIV_BROADCAST_2D_1D_RULE = "rextio-torch/tensor-div-f32-cpu-2d-1d-broadcast"
MATMUL_BINOP_RULE = "rextio-torch/tensor-matmul-f32-cpu-2d"
MATMUL_CALL_RULE = "rextio-torch/tensor-matmul-call-f32-cpu-2d"
MATMUL_BINOP_MIXED_RANK_RULE = "rextio-torch/tensor-matmul-f32-cpu-mixed-rank"
MATMUL_CALL_MIXED_RANK_RULE = "rextio-torch/tensor-matmul-call-f32-cpu-mixed-rank"
MATMUL_CALL_TARGET = "torch.matmul"
FUNCTION_ADD_RULE = "rextio-torch/function-add-f32-cpu-rank1-2"
FUNCTION_SUB_RULE = "rextio-torch/function-sub-f32-cpu-rank1-2"
FUNCTION_MUL_RULE = "rextio-torch/function-mul-f32-cpu-rank1-2"
FUNCTION_DIV_RULE = "rextio-torch/function-div-f32-cpu-rank1-2"

ADD_RULES: frozenset[str] = frozenset({ADD_SAME_RANK_RULE, ADD_BROADCAST_2D_1D_RULE})
MUL_RULES: frozenset[str] = frozenset({MUL_SAME_RANK_RULE, MUL_BROADCAST_2D_1D_RULE})
SUB_RULES: frozenset[str] = frozenset({SUB_SAME_RANK_RULE, SUB_BROADCAST_2D_1D_RULE})
DIV_RULES: frozenset[str] = frozenset({DIV_SAME_RANK_RULE, DIV_BROADCAST_2D_1D_RULE})

_F32_TENSOR_TYPES: frozenset[str] = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})
_FUNCTION_ELEMENTWISE: dict[str, tuple[str, str]] = {
    "torch.add": (FUNCTION_ADD_RULE, DIAGNOSTIC_FUNCTION_ADD),
    "torch.sub": (FUNCTION_SUB_RULE, DIAGNOSTIC_FUNCTION_SUB),
    "torch.mul": (FUNCTION_MUL_RULE, DIAGNOSTIC_FUNCTION_MUL),
    "torch.div": (FUNCTION_DIV_RULE, DIAGNOSTIC_FUNCTION_DIV),
}


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def _elementwise_result_type(left: str, right: str) -> str | None:
    """Return the existing operator matrix result for two float32 CPU tensors."""
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        return None
    if left == right:
        return left
    if {left, right} == _F32_TENSOR_TYPES:
        return TENSOR_F32_CPU_2D
    return None


def _try_claim_functional_elementwise(site: ClaimSite) -> ClaimResult | None:
    if site.kind != "call" or site.target not in _FUNCTION_ELEMENTWISE:
        return None
    rule_id, diagnostic = _FUNCTION_ELEMENTWISE[site.target]
    if (
        site.receiver is not None
        or site.keywords
        or len(site.operand_types) != 2
        or len(site.operand_literals) != 2
        or any(literal.is_literal for literal in site.operand_literals)
    ):
        return reject(
            site,
            diagnostic,
            "functional elementwise calls require exactly two positional tensor operands",
            (
                f"Call {site.target}(a, b) with two float32 CPU tensors; "
                "omit scalar operands, alpha/out/rounding_mode, and every keyword."
            ),
        )
    left, right = site.operand_types
    if left is None or right is None:
        return NotCovered()
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU rank-1/2 tensor surface",
            "Annotate both operands with TensorF32Cpu1D or TensorF32Cpu2D.",
        )
    result_type = _elementwise_result_type(left, right)
    if result_type is None:
        return reject(
            site,
            diagnostic,
            f"unsupported functional elementwise operand types {left!r} and {right!r}",
            (
                "Use same-rank float32 CPU tensors or the existing rank-2/rank-1 "
                "trailing-broadcast matrix."
            ),
        )
    return Claimed(rule_id=rule_id, result_type=result_type)


def _try_claim_add(site: ClaimSite) -> ClaimResult | None:
    if site.kind != "binop" or site.target != "+":
        return None
    if site.receiver is not None or site.keywords:
        return reject(
            site,
            DIAGNOSTIC_ADD,
            "only binary + between two tensor operands is supported",
            "Write a + b with two float32 CPU tensors; no method or keyword forms.",
        )
    if len(site.operand_types) != 2:
        return reject(
            site,
            DIAGNOSTIC_ADD,
            "elementwise + requires exactly two operands",
            "Write a + b with two annotated tensor operands.",
        )
    left, right = site.operand_types
    if left is None or right is None:
        return NotCovered()
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU rank-1/2 tensor surface",
            "Annotate both operands with TensorF32Cpu1D or TensorF32Cpu2D.",
        )
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f"+ requires float32 CPU rank-1/2 operands; got {left!r}, {right!r}",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for both add operands.",
        )
    if left == right:
        return Claimed(rule_id=ADD_SAME_RANK_RULE, result_type=left)
    # Trailing bias broadcast: rank-2 + rank-1 (either order) → rank-2.
    if {left, right} == _F32_TENSOR_TYPES:
        return Claimed(rule_id=ADD_BROADCAST_2D_1D_RULE, result_type=TENSOR_F32_CPU_2D)
    return reject(
        site,
        DIAGNOSTIC_ADD,
        f"unsupported + operand types {left!r} and {right!r}",
        "Supported forms: same-rank + and rank-2 + rank-1 trailing bias broadcast.",
    )


def _try_claim_mul(site: ClaimSite) -> ClaimResult | None:
    if site.kind != "binop" or site.target != "*":
        return None
    if site.receiver is not None or site.keywords:
        return reject(
            site,
            DIAGNOSTIC_MUL,
            "only binary * between two tensor operands is supported",
            "Write a * b with two float32 CPU tensors; no method or keyword forms.",
        )
    if len(site.operand_types) != 2:
        return reject(
            site,
            DIAGNOSTIC_MUL,
            "elementwise * requires exactly two operands",
            "Write a * b with two annotated tensor operands.",
        )
    left, right = site.operand_types
    if left is None or right is None:
        return NotCovered()
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU rank-1/2 tensor surface",
            "Annotate both operands with TensorF32Cpu1D or TensorF32Cpu2D.",
        )
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f"* requires float32 CPU rank-1/2 operands; got {left!r}, {right!r}",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for both multiply operands.",
        )
    if left == right:
        return Claimed(rule_id=MUL_SAME_RANK_RULE, result_type=left)
    if {left, right} == _F32_TENSOR_TYPES:
        return Claimed(rule_id=MUL_BROADCAST_2D_1D_RULE, result_type=TENSOR_F32_CPU_2D)
    return reject(
        site,
        DIAGNOSTIC_MUL,
        f"unsupported * operand types {left!r} and {right!r}",
        "Supported forms: same-rank * and rank-2 * rank-1 trailing bias broadcast.",
    )


def _try_claim_sub(site: ClaimSite) -> ClaimResult | None:
    if site.kind != "binop" or site.target != "-":
        return None
    if site.receiver is not None or site.keywords:
        return reject(
            site,
            DIAGNOSTIC_SUB,
            "only binary - between two tensor operands is supported",
            "Write a - b with two float32 CPU tensors; no method or keyword forms.",
        )
    if len(site.operand_types) != 2:
        return reject(
            site,
            DIAGNOSTIC_SUB,
            "elementwise - requires exactly two operands",
            "Write a - b with two annotated tensor operands.",
        )
    left, right = site.operand_types
    if left is None or right is None:
        return NotCovered()
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU rank-1/2 tensor surface",
            "Annotate both operands with TensorF32Cpu1D or TensorF32Cpu2D.",
        )
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f"- requires float32 CPU rank-1/2 operands; got {left!r}, {right!r}",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for both subtraction operands.",
        )
    if left == right:
        return Claimed(rule_id=SUB_SAME_RANK_RULE, result_type=left)
    if {left, right} == _F32_TENSOR_TYPES:
        return Claimed(rule_id=SUB_BROADCAST_2D_1D_RULE, result_type=TENSOR_F32_CPU_2D)
    return reject(
        site,
        DIAGNOSTIC_SUB,
        f"unsupported - operand types {left!r} and {right!r}",
        "Supported forms: same-rank - and rank-2 - rank-1 trailing broadcast.",
    )


def _try_claim_div(site: ClaimSite) -> ClaimResult | None:
    if site.kind != "binop" or site.target != "/":
        return None
    if site.receiver is not None or site.keywords:
        return reject(
            site,
            DIAGNOSTIC_DIV,
            "only binary / between two tensor operands is supported",
            "Write a / b with two float32 CPU tensors; no method or keyword forms.",
        )
    if len(site.operand_types) != 2:
        return reject(
            site,
            DIAGNOSTIC_DIV,
            "elementwise / requires exactly two operands",
            "Write a / b with two annotated tensor operands.",
        )
    left, right = site.operand_types
    if left is None or right is None:
        return NotCovered()
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU rank-1/2 tensor surface",
            "Annotate both operands with TensorF32Cpu1D or TensorF32Cpu2D.",
        )
    if left not in _F32_TENSOR_TYPES or right not in _F32_TENSOR_TYPES:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f"/ requires float32 CPU rank-1/2 operands; got {left!r}, {right!r}",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for both division operands.",
        )
    if left == right:
        return Claimed(rule_id=DIV_SAME_RANK_RULE, result_type=left)
    if {left, right} == _F32_TENSOR_TYPES:
        return Claimed(rule_id=DIV_BROADCAST_2D_1D_RULE, result_type=TENSOR_F32_CPU_2D)
    return reject(
        site,
        DIAGNOSTIC_DIV,
        f"unsupported / operand types {left!r} and {right!r}",
        "Supported forms: same-rank / and rank-2 / rank-1 trailing broadcast.",
    )


def _try_claim_matmul_binop(site: ClaimSite) -> ClaimResult | None:
    if site.kind != "binop" or site.target != "@":
        return None
    if site.receiver is not None or site.keywords:
        return reject(
            site,
            DIAGNOSTIC_MATMUL,
            "only binary @ between supported rank-1/rank-2 tensors is supported",
            "Write a @ b with rank-2/rank-2 or mixed rank-2/rank-1 operands.",
        )
    return _claim_matmul_operands(
        site,
        rank2_rule_id=MATMUL_BINOP_RULE,
        mixed_rule_id=MATMUL_BINOP_MIXED_RANK_RULE,
        diagnostic=DIAGNOSTIC_MATMUL,
        mixed_diagnostic=DIAGNOSTIC_MATMUL_MIXED_RANK,
    )


def _try_claim_matmul_call(site: ClaimSite) -> ClaimResult | None:
    if site.kind != "call":
        return None
    target = site.target
    method = _method_name(target)
    if target == MATMUL_CALL_TARGET:
        if site.receiver is not None:
            return NotCovered()
        if site.keywords:
            return reject(
                site,
                DIAGNOSTIC_MATMUL_CALL,
                "torch.matmul supports only two positional tensor operands",
                "Pass both tensors positionally; omit keywords.",
            )
        return _claim_matmul_operands(
            site,
            rank2_rule_id=MATMUL_CALL_RULE,
            mixed_rule_id=MATMUL_CALL_MIXED_RANK_RULE,
            diagnostic=DIAGNOSTIC_MATMUL_CALL,
            mixed_diagnostic=DIAGNOSTIC_MATMUL_CALL_MIXED_RANK,
        )
    if method == "matmul":
        # Method form: receiver @ operand.
        if site.receiver is None:
            return NotCovered()
        if site.keywords or len(site.operand_types) != 1:
            return reject(
                site,
                DIAGNOSTIC_MATMUL_CALL,
                "only .matmul(other) with one positional tensor is supported",
                "Call .matmul(other) with one rank-1/rank-2 tensor argument.",
            )
        receiver_type = site.receiver.arg_type
        other = site.operand_types[0]
        if receiver_type is None or other is None:
            return NotCovered()
        return _claim_matmul_pair(
            site,
            receiver_type,
            other,
            rank2_rule_id=MATMUL_CALL_RULE,
            mixed_rule_id=MATMUL_CALL_MIXED_RANK_RULE,
            diagnostic=DIAGNOSTIC_MATMUL_CALL,
            mixed_diagnostic=DIAGNOSTIC_MATMUL_CALL_MIXED_RANK,
        )
    return None


def _claim_matmul_operands(
    site: ClaimSite,
    *,
    rank2_rule_id: str,
    mixed_rule_id: str,
    diagnostic: str,
    mixed_diagnostic: str,
) -> ClaimResult:
    if len(site.operand_types) != 2:
        return reject(
            site,
            diagnostic,
            "matmul requires exactly two positional tensor operands",
            "Pass two supported rank-1/rank-2 tensor operands.",
        )
    left, right = site.operand_types
    if left is None or right is None:
        return NotCovered()
    return _claim_matmul_pair(
        site,
        left,
        right,
        rank2_rule_id=rank2_rule_id,
        mixed_rule_id=mixed_rule_id,
        diagnostic=diagnostic,
        mixed_diagnostic=mixed_diagnostic,
    )


def _claim_matmul_pair(
    site: ClaimSite,
    left: str,
    right: str,
    *,
    rank2_rule_id: str,
    mixed_rule_id: str,
    diagnostic: str,
    mixed_diagnostic: str,
) -> ClaimResult:
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU rank-1/2 tensor surface",
            "Annotate matmul operands as TensorF32Cpu1D or TensorF32Cpu2D.",
        )
    if (left, right) == (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D):
        return Claimed(rule_id=rank2_rule_id, result_type=TENSOR_F32_CPU_2D)
    if (left, right) in {
        (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
        (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D),
    }:
        return Claimed(rule_id=mixed_rule_id, result_type=TENSOR_F32_CPU_1D)
    return reject(
        site,
        mixed_diagnostic if TENSOR_F32_CPU_1D in {left, right} else diagnostic,
        f"unsupported matmul operand ranks for {left!r} and {right!r}",
        (
            "Use rank-2 × rank-2 for a rank-2 result, or rank-2 × rank-1 / "
            "rank-1 × rank-2 for a rank-1 result. Rank-1 × rank-1 would produce "
            "an unregistered rank-0 result."
        )
    )


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim supported tensor binops/calls, else None when not this lane."""
    for lane in (
        _try_claim_functional_elementwise,
        _try_claim_add,
        _try_claim_mul,
        _try_claim_sub,
        _try_claim_div,
        _try_claim_matmul_binop,
        _try_claim_matmul_call,
    ):
        result = lane(site)
        if result is not None:
            return result
    return None


__all__ = [
    "ADD_BROADCAST_2D_1D_RULE",
    "ADD_RULES",
    "ADD_SAME_RANK_RULE",
    "DIV_BROADCAST_2D_1D_RULE",
    "DIV_RULES",
    "DIV_SAME_RANK_RULE",
    "FUNCTION_ADD_RULE",
    "FUNCTION_DIV_RULE",
    "FUNCTION_MUL_RULE",
    "FUNCTION_SUB_RULE",
    "MATMUL_BINOP_MIXED_RANK_RULE",
    "MATMUL_BINOP_RULE",
    "MATMUL_CALL_MIXED_RANK_RULE",
    "MATMUL_CALL_RULE",
    "MATMUL_CALL_TARGET",
    "MUL_BROADCAST_2D_1D_RULE",
    "MUL_RULES",
    "MUL_SAME_RANK_RULE",
    "SUB_BROADCAST_2D_1D_RULE",
    "SUB_RULES",
    "SUB_SAME_RANK_RULE",
    "try_claim",
]
