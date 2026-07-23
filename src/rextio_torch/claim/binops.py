"""Fail-closed claims for elementwise binary ops and rank-2 matmul."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_ADD,
    DIAGNOSTIC_DIV,
    DIAGNOSTIC_MUL,
    DIAGNOSTIC_MATMUL,
    DIAGNOSTIC_MATMUL_CALL,
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
MATMUL_CALL_TARGET = "torch.matmul"

ADD_RULES: frozenset[str] = frozenset({ADD_SAME_RANK_RULE, ADD_BROADCAST_2D_1D_RULE})
MUL_RULES: frozenset[str] = frozenset({MUL_SAME_RANK_RULE, MUL_BROADCAST_2D_1D_RULE})
SUB_RULES: frozenset[str] = frozenset({SUB_SAME_RANK_RULE, SUB_BROADCAST_2D_1D_RULE})
DIV_RULES: frozenset[str] = frozenset({DIV_SAME_RANK_RULE, DIV_BROADCAST_2D_1D_RULE})

_F32_TENSOR_TYPES: frozenset[str] = frozenset({TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D})


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


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
            "only binary @ between two rank-2 tensors is supported",
            "Write a @ b with two TensorF32Cpu2D operands.",
        )
    return _claim_matmul_operands(
        site,
        rule_id=MATMUL_BINOP_RULE,
        diagnostic=DIAGNOSTIC_MATMUL,
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
            rule_id=MATMUL_CALL_RULE,
            diagnostic=DIAGNOSTIC_MATMUL_CALL,
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
                "Call .matmul(other) with a single TensorF32Cpu2D argument.",
            )
        receiver_type = site.receiver.arg_type
        other = site.operand_types[0]
        if receiver_type is None or other is None:
            return NotCovered()
        return _claim_matmul_pair(
            site,
            receiver_type,
            other,
            rule_id=MATMUL_CALL_RULE,
            diagnostic=DIAGNOSTIC_MATMUL_CALL,
        )
    return None


def _claim_matmul_operands(
    site: ClaimSite,
    *,
    rule_id: str,
    diagnostic: str,
) -> ClaimResult:
    if len(site.operand_types) != 2:
        return reject(
            site,
            diagnostic,
            "matmul requires exactly two rank-2 tensor operands",
            "Pass two TensorF32Cpu2D operands.",
        )
    left, right = site.operand_types
    if left is None or right is None:
        return NotCovered()
    return _claim_matmul_pair(
        site,
        left,
        right,
        rule_id=rule_id,
        diagnostic=diagnostic,
    )


def _claim_matmul_pair(
    site: ClaimSite,
    left: str,
    right: str,
    *,
    rule_id: str,
    diagnostic: str,
) -> ClaimResult:
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU rank-1/2 tensor surface",
            "Annotate matmul operands as TensorF32Cpu2D.",
        )
    if left != TENSOR_F32_CPU_2D or right != TENSOR_F32_CPU_2D:
        return reject(
            site,
            diagnostic,
            f"rank-2 matmul requires float32 CPU rank-2 operands; got {left!r}, {right!r}",
            "Use TensorF32Cpu2D for both matmul operands.",
        )
    return Claimed(rule_id=rule_id, result_type=TENSOR_F32_CPU_2D)


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim supported tensor binops/calls, else None when not this lane."""
    for lane in (
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
    "MATMUL_BINOP_RULE",
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
