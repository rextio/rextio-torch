"""Fail-closed static-dimension claims for bounded ``mean`` / ``sum`` forms."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_MEAN,
    DIAGNOSTIC_SUM,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

MEAN_RULE = "rextio-torch/tensor-mean-dim1-f32-cpu-2d"
SUM_RULE = "rextio-torch/tensor-sum-dim1-f32-cpu-2d"
MEAN_STATIC_RULE = "rextio-torch/mean-static-dim-f32-cpu-rank1-2"
SUM_STATIC_RULE = "rextio-torch/sum-static-dim-f32-cpu-rank1-2"

_REDUCTION_RULES: dict[str, tuple[str, str, str]] = {
    "mean": (MEAN_RULE, MEAN_STATIC_RULE, DIAGNOSTIC_MEAN),
    "sum": (SUM_RULE, SUM_STATIC_RULE, DIAGNOSTIC_SUM),
}

_FUNCTION_TARGETS: dict[str, str] = {
    "torch.mean": "mean",
    "torch.sum": "sum",
}

REDUCTION_RULES: frozenset[str] = frozenset(
    {MEAN_RULE, SUM_RULE, MEAN_STATIC_RULE, SUM_STATIC_RULE}
)


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def _keyword_map(site: ClaimSite) -> dict[str, object] | None:
    """Return a duplicate-free name→literal value map, else ``None``."""
    values: dict[str, object] = {}
    expected_types = {"dim": "int", "keepdim": "bool"}
    for keyword in site.keywords:
        if (
            keyword.name in values
            or not keyword.literal.is_literal
            or (
                keyword.name in expected_types
                and keyword.arg_type != expected_types[keyword.name]
            )
        ):
            return None
        values[keyword.name] = keyword.literal.value
    return values


def _input_and_positional_dim(
    site: ClaimSite,
) -> tuple[str | None, object | None, bool] | None:
    """Resolve input type and an optional aligned positional literal dimension."""
    receiver = site.receiver
    if receiver is None:
        if _FUNCTION_TARGETS.get(site.target) is None:
            return None
        base_arity = 1
        if len(site.operand_types) not in {base_arity, base_arity + 1}:
            return (None, None, False)
        input_type = site.operand_types[0]
    else:
        base_arity = 0
        if len(site.operand_types) not in {base_arity, base_arity + 1}:
            return (None, None, False)
        input_type = receiver.arg_type

    has_positional_dim = len(site.operand_types) == base_arity + 1
    if not has_positional_dim:
        return (input_type, None, False)
    if len(site.operand_literals) != len(site.operand_types):
        return (None, None, False)
    dim_index = base_arity
    if site.operand_types[dim_index] != "int":
        return (None, None, False)
    literal = site.operand_literals[dim_index]
    if not literal.is_literal:
        return (None, None, False)
    return (input_type, literal.value, True)


def _result_type(input_type: str, dim: int, keepdim: bool) -> str | None:
    if input_type == TENSOR_F32_CPU_2D and dim in {0, 1}:
        return TENSOR_F32_CPU_2D if keepdim else TENSOR_F32_CPU_1D
    if input_type == TENSOR_F32_CPU_1D and dim == 0 and keepdim:
        return TENSOR_F32_CPU_1D
    return None


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim representable static-dimension mean/sum method and function forms."""
    method = _method_name(site.target)
    if site.kind != "call" or method not in _REDUCTION_RULES:
        return None
    if site.receiver is None and _FUNCTION_TARGETS.get(site.target) != method:
        return NotCovered()
    legacy_rule, static_rule, diagnostic = _REDUCTION_RULES[method]

    resolved = _input_and_positional_dim(site)
    if resolved is None:
        return NotCovered()
    input_type, positional_dim, has_positional_dim = resolved
    if input_type is None:
        # Distinguish an unresolved canonical one-input form from malformed arity.
        canonical_arity = 1 if site.receiver is None else 0
        if len(site.operand_types) == canonical_arity:
            return NotCovered()
        return reject(
            site,
            diagnostic,
            f"{method} accepts one tensor input/receiver and at most one positional literal dim",
            "Pass only a tensor plus optional literal dim; keepdim must be a named literal.",
        )

    keywords = _keyword_map(site)
    if keywords is None:
        return reject(
            site,
            diagnostic,
            f"{method} keywords must be unique static literals",
            "Use a literal dim and optional named literal keepdim; omit dtype/out options.",
        )
    allowed_keywords = {"keepdim"} if has_positional_dim else {"dim", "keepdim"}
    if not set(keywords) <= allowed_keywords:
        return reject(
            site,
            diagnostic,
            f"unsupported or duplicate positional/keyword options for {method}",
            "Provide dim once (positionally or by keyword) and optional named keepdim only.",
        )
    if has_positional_dim:
        dim = positional_dim
    else:
        if "dim" not in keywords:
            return reject(
                site,
                diagnostic,
                f"{method} requires a literal dim=0 or dim=1",
                "Provide dim as one positional literal or the named literal keyword.",
            )
        dim = keywords["dim"]
    keepdim = keywords.get("keepdim", False)

    if not isinstance(dim, int) or isinstance(dim, bool) or dim not in {0, 1}:
        return reject(
            site,
            diagnostic,
            f"{method} requires literal dim=0 or dim=1; got dim={dim!r}",
            "Use the literal dimension 0 or 1.",
        )
    if not isinstance(keepdim, bool):
        return reject(
            site,
            diagnostic,
            f"{method} keepdim must be a bool literal; got {keepdim!r}",
            "Omit keepdim for its False default or use keepdim=True/False.",
        )
    if not is_tensor_type(input_type):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "input type is outside the float32 CPU tensor surface",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for the reduction input.",
        )
    result_type = _result_type(input_type, dim, keepdim)
    if result_type is None:
        return reject(
            site,
            diagnostic,
            (
                f"{method}(dim={dim}, keepdim={keepdim}) on {input_type!r} "
                "does not have an already-registered rank-1/rank-2 result type"
            ),
            "Keep the reduction result within the registered float32 CPU rank-1/rank-2 types.",
        )

    is_legacy = (
        site.receiver is not None
        and not has_positional_dim
        and input_type == TENSOR_F32_CPU_2D
        and dim == 1
        and keepdim is False
        and set(keywords) == {"dim", "keepdim"}
    )
    return Claimed(
        rule_id=legacy_rule if is_legacy else static_rule,
        result_type=result_type,
    )


__all__ = [
    "MEAN_RULE",
    "MEAN_STATIC_RULE",
    "REDUCTION_RULES",
    "SUM_RULE",
    "SUM_STATIC_RULE",
    "try_claim",
]
