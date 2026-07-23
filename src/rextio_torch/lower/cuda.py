"""Authorization-bound lowering for the build-only CUDA E2 slice."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_torch.claim.cuda import (
    CUDA_BIAS_ADD_RULE,
    CUDA_MATMUL_RULE,
    CUDA_MEAN_DIM1_RULE,
    CUDA_RELU_RULE,
    CUDA_RULES,
)
from rextio_torch.diagnostics import TENSOR_F32_CUDA0_1D, TENSOR_F32_CUDA0_2D
from rextio_torch.plugin_types import plugin_type
from rextio_torch.rust_snippets import (
    ADD,
    MATMUL,
    MEAN_DIM1_KEEPFALSE,
    RELU,
    add_helper,
    boundary_helpers,
    matmul_helper,
    mean_dim1_keepfalse_helper,
    relu_helper,
)

CUDA_PROVIDER_ID = "rextio-device-cuda"
CUDA_CAPABILITY_ID = "cuda-libtorch-linux-x86_64"


def _require_authorization(ctx: LoweringContext, result_type: str) -> None:
    authorization = ctx.device_authorization
    try:
        metadata = plugin_type(result_type).device_value_metadata
    except KeyError:
        metadata = None
    if (
        authorization is None
        or authorization.provider_id != CUDA_PROVIDER_ID
        or authorization.capability_id != CUDA_CAPABILITY_ID
        or metadata is None
        or not authorization.authorizes(metadata)
    ):
        raise ValueError(
            "rextio-torch CUDA lowering requires exact authorization from "
            "rextio-device-cuda/cuda-libtorch-linux-x86_64"
        )


def _keyword_values(claimed: ClaimSite) -> dict[str, object]:
    values: dict[str, object] = {}
    for keyword in claimed.keywords:
        if (
            keyword.name in values
            or keyword.name not in {"dim", "keepdim"}
            or not keyword.literal.is_literal
            or (keyword.name == "dim" and keyword.arg_type != "int")
            or (keyword.name == "keepdim" and keyword.arg_type != "bool")
        ):
            raise ValueError("rextio-torch CUDA mean requires unique literal keywords")
        values[keyword.name] = keyword.literal.value
    return values


def _nonliteral_tensor_operands(claimed: ClaimSite) -> bool:
    return (
        not claimed.operand_literals
        or (
            len(claimed.operand_literals) == len(claimed.operand_types)
            and not any(item.is_literal for item in claimed.operand_literals)
        )
    )


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower one exact CUDA rule after independent metadata validation."""
    if claimed.rule_id not in CUDA_RULES:
        return None
    if ctx.backend != "pyo3":
        raise ValueError("rextio-torch CUDA E2 supports PyO3 host extensions only")
    _require_authorization(ctx, claimed.result_type or "")

    if claimed.rule_id == CUDA_MATMUL_RULE:
        if (
            claimed.kind != "binop"
            or claimed.target != "@"
            or claimed.operand_types
            != (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D)
            or not _nonliteral_tensor_operands(claimed)
            or claimed.result_type != TENSOR_F32_CUDA0_2D
            or claimed.receiver is not None
            or claimed.keywords
            or len(ctx.operands) != 2
            or ctx.receiver is not None
        ):
            raise ValueError("rextio-torch CUDA matmul metadata changed after claim")
        left, right = ctx.operands
        return LoweredExpr(
            rust=f"{MATMUL}(&{left}, &{right})?",
            helpers=(boundary_helpers(), matmul_helper()),
        )

    if claimed.rule_id == CUDA_BIAS_ADD_RULE:
        if (
            claimed.kind != "binop"
            or claimed.target != "+"
            or claimed.operand_types
            != (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_1D)
            or not _nonliteral_tensor_operands(claimed)
            or claimed.result_type != TENSOR_F32_CUDA0_2D
            or claimed.receiver is not None
            or claimed.keywords
            or len(ctx.operands) != 2
            or ctx.receiver is not None
        ):
            raise ValueError("rextio-torch CUDA bias-add metadata changed after claim")
        matrix, bias = ctx.operands
        return LoweredExpr(
            rust=f"{ADD}(&{matrix}, &{bias})?",
            helpers=(boundary_helpers(), add_helper()),
        )

    if claimed.rule_id == CUDA_RELU_RULE:
        if (
            claimed.kind != "call"
            or claimed.target.rpartition(".")[2] != "relu"
            or claimed.receiver is None
            or claimed.receiver.arg_type != TENSOR_F32_CUDA0_2D
            or claimed.result_type != TENSOR_F32_CUDA0_2D
            or claimed.operand_types
            or claimed.operand_literals
            or claimed.keywords
            or ctx.operands
            or ctx.receiver is None
        ):
            raise ValueError("rextio-torch CUDA ReLU metadata changed after claim")
        return LoweredExpr(
            rust=f"{RELU}(&{ctx.receiver})?",
            helpers=(boundary_helpers(), relu_helper()),
        )

    if claimed.rule_id == CUDA_MEAN_DIM1_RULE:
        values = _keyword_values(claimed)
        if (
            claimed.kind != "call"
            or claimed.target.rpartition(".")[2] != "mean"
            or claimed.receiver is None
            or claimed.receiver.arg_type != TENSOR_F32_CUDA0_2D
            or claimed.result_type != TENSOR_F32_CUDA0_1D
            or claimed.operand_types
            or claimed.operand_literals
            or set(values) not in ({"dim"}, {"dim", "keepdim"})
            or type(values.get("dim")) is not int
            or values.get("dim") != 1
            or type(values.get("keepdim", False)) is not bool
            or values.get("keepdim", False) is not False
            or ctx.operands
            or ctx.receiver is None
        ):
            raise ValueError("rextio-torch CUDA mean metadata changed after claim")
        return LoweredExpr(
            rust=f"{MEAN_DIM1_KEEPFALSE}(&{ctx.receiver})?",
            helpers=(boundary_helpers(), mean_dim1_keepfalse_helper()),
        )

    raise ValueError(f"unexpected rextio-torch CUDA rule: {claimed.rule_id!r}")


__all__ = ["CUDA_CAPABILITY_ID", "CUDA_PROVIDER_ID", "try_lower"]
