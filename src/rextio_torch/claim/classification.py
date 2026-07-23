"""Fail-closed claims for representable static softmax/argmax forms."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_ARGMAX,
    DIAGNOSTIC_SOFTMAX,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
    is_tensor_type,
    reject,
)

SOFTMAX_RULE = "rextio-torch/tensor-softmax-dim1-f32-cpu-2d"
ARGMAX_RULE = "rextio-torch/tensor-argmax-dim1-keepfalse-i64-cpu-1d"
SOFTMAX_STATIC_RULE = "rextio-torch/softmax-static-dim-f32-cpu-rank1-2"
ARGMAX_STATIC_RULE = "rextio-torch/argmax-static-dim-i64-cpu-rank1"
CLASSIFICATION_RULES: frozenset[str] = frozenset(
    {SOFTMAX_RULE, ARGMAX_RULE, SOFTMAX_STATIC_RULE, ARGMAX_STATIC_RULE}
)

_RULES: dict[str, tuple[str, str, str]] = {
    "softmax": (SOFTMAX_RULE, SOFTMAX_STATIC_RULE, DIAGNOSTIC_SOFTMAX),
    "argmax": (ARGMAX_RULE, ARGMAX_STATIC_RULE, DIAGNOSTIC_ARGMAX),
}
_FUNCTION_TARGETS: dict[str, str] = {
    "torch.softmax": "softmax",
    "torch.argmax": "argmax",
}


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def _keyword_map(site: ClaimSite) -> dict[str, object] | None:
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
    receiver = site.receiver
    if receiver is None:
        if _FUNCTION_TARGETS.get(site.target) is None:
            return None
        base_arity = 1
        if len(site.operand_types) not in {1, 2}:
            return (None, None, False)
        input_type = site.operand_types[0]
    else:
        base_arity = 0
        if len(site.operand_types) not in {0, 1}:
            return (None, None, False)
        input_type = receiver.arg_type

    has_positional_dim = len(site.operand_types) == base_arity + 1
    if not has_positional_dim:
        return (input_type, None, False)
    if len(site.operand_literals) != len(site.operand_types):
        return (None, None, False)
    dim_index = base_arity
    literal = site.operand_literals[dim_index]
    if site.operand_types[dim_index] != "int" or not literal.is_literal:
        return (None, None, False)
    return (input_type, literal.value, True)


def _softmax_result(input_type: str, dim: int) -> str | None:
    if input_type == TENSOR_F32_CPU_1D and dim == 0:
        return TENSOR_F32_CPU_1D
    if input_type == TENSOR_F32_CPU_2D and dim in {0, 1}:
        return TENSOR_F32_CPU_2D
    return None


def _argmax_result(input_type: str, dim: int, keepdim: bool) -> str | None:
    if input_type == TENSOR_F32_CPU_2D and dim in {0, 1} and not keepdim:
        return TENSOR_I64_CPU_1D
    if input_type == TENSOR_F32_CPU_1D and dim == 0 and keepdim:
        return TENSOR_I64_CPU_1D
    return None


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim only classification forms whose exact output type is registered."""
    method = _method_name(site.target)
    if site.kind != "call" or method not in _RULES:
        return None
    if site.receiver is None and _FUNCTION_TARGETS.get(site.target) != method:
        return NotCovered()
    legacy_rule, static_rule, diagnostic = _RULES[method]

    resolved = _input_and_positional_dim(site)
    if resolved is None:
        return NotCovered()
    input_type, positional_dim, has_positional_dim = resolved
    if input_type is None:
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
            "Use a literal dim and, for argmax only, optional named literal keepdim.",
        )
    allowed = set() if has_positional_dim else {"dim"}
    if method == "argmax":
        allowed.add("keepdim")
    if not set(keywords) <= allowed:
        return reject(
            site,
            diagnostic,
            f"unsupported or duplicate positional/keyword options for {method}",
            "Provide dim once; omit dtype and use keepdim only as an argmax keyword.",
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
    if not isinstance(dim, int) or isinstance(dim, bool) or dim not in {0, 1}:
        return reject(
            site,
            diagnostic,
            f"{method} requires literal dim=0 or dim=1; got dim={dim!r}",
            "Use the literal dimension 0 or 1.",
        )
    keepdim = keywords.get("keepdim", False)
    if not isinstance(keepdim, bool):
        return reject(
            site,
            diagnostic,
            f"argmax keepdim must be a bool literal; got {keepdim!r}",
            "Omit keepdim for False or use the named literal keepdim=True/False.",
        )
    if not is_tensor_type(input_type):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "input type is outside the float32 CPU tensor surface",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for classification input.",
        )

    result_type = (
        _softmax_result(input_type, dim)
        if method == "softmax"
        else _argmax_result(input_type, dim, keepdim)
    )
    if result_type is None:
        detail = (
            f"{method}(dim={dim})"
            if method == "softmax"
            else f"argmax(dim={dim}, keepdim={keepdim})"
        )
        return reject(
            site,
            diagnostic,
            (
                f"{detail} on {input_type!r} does not have an existing exact "
                "classification result type"
            ),
            (
                "Keep argmax outputs at registered int64 CPU rank-1; rank-2 "
                "keepdim=True and rank-1 keepdim=False remain fallback."
            ),
        )

    is_legacy = (
        site.receiver is not None
        and not has_positional_dim
        and input_type == TENSOR_F32_CPU_2D
        and dim == 1
        and (
            (method == "softmax" and set(keywords) == {"dim"})
            or (
                method == "argmax"
                and keepdim is False
                and set(keywords) == {"dim", "keepdim"}
            )
        )
    )
    return Claimed(
        rule_id=legacy_rule if is_legacy else static_rule,
        result_type=result_type,
    )


__all__ = [
    "ARGMAX_RULE",
    "ARGMAX_STATIC_RULE",
    "CLASSIFICATION_RULES",
    "SOFTMAX_RULE",
    "SOFTMAX_STATIC_RULE",
    "try_claim",
]
