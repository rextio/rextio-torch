"""Fail-closed claims for the rank-2 CPU classification-head method chain.

Only ``.softmax(dim=1)`` and ``.argmax(dim=1, keepdim=False)`` are
recognized.  The latter returns the dedicated int64 rank-1 boundary type.
Functional spellings and dtype overrides deliberately stay outside the slice.
"""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_torch.diagnostics import (
    DIAGNOSTIC_ARGMAX,
    DIAGNOSTIC_SOFTMAX,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
    is_tensor_type,
    reject,
)

SOFTMAX_RULE = "rextio-torch/tensor-softmax-dim1-f32-cpu-2d"
ARGMAX_RULE = "rextio-torch/tensor-argmax-dim1-keepfalse-i64-cpu-1d"
CLASSIFICATION_RULES: frozenset[str] = frozenset({SOFTMAX_RULE, ARGMAX_RULE})

_RULES: dict[str, tuple[str, str, frozenset[str], str]] = {
    "softmax": (SOFTMAX_RULE, DIAGNOSTIC_SOFTMAX, frozenset({"dim"}), TENSOR_F32_CPU_2D),
    "argmax": (
        ARGMAX_RULE,
        DIAGNOSTIC_ARGMAX,
        frozenset({"dim", "keepdim"}),
        TENSOR_I64_CPU_1D,
    ),
}


def _method_name(target: str) -> str:
    return target.rpartition(".")[2]


def _keyword_map(site: ClaimSite) -> dict[str, object] | None:
    values: dict[str, object] = {}
    for keyword in site.keywords:
        if not keyword.literal.is_literal:
            return None
        values[keyword.name] = keyword.literal.value
    return values


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim the literal-only method forms that make up the classification head."""
    method = _method_name(site.target)
    if site.kind != "call" or method not in _RULES:
        return None
    rule_id, diagnostic, expected_keywords, result_type = _RULES[method]
    receiver = site.receiver
    if receiver is None:
        # ``torch.softmax`` / ``torch.argmax`` and functional variants are excluded.
        return NotCovered()
    if site.operand_types:
        return reject(
            site,
            diagnostic,
            f"positional arguments to .{method} are not supported",
            "Use the documented literal keyword-only classification-head form.",
        )
    keywords = _keyword_map(site)
    if keywords is None or set(keywords) != expected_keywords:
        required = ".softmax(dim=1)" if method == "softmax" else ".argmax(dim=1, keepdim=False)"
        return reject(
            site,
            diagnostic,
            f"only {required} with literal keywords is supported",
            f"Use {required}; dtype overrides and dynamic values are unsupported.",
        )
    dim = keywords["dim"]
    if not isinstance(dim, int) or isinstance(dim, bool) or dim != 1:
        return reject(
            site,
            diagnostic,
            f"{method} requires dim=1 literal; got dim={dim!r}",
            "Use the literal keyword dim=1.",
        )
    if method == "argmax" and keywords["keepdim"] is not False:
        return reject(
            site,
            diagnostic,
            f"argmax requires keepdim=False; got keepdim={keywords['keepdim']!r}",
            "Use the literal keyword keepdim=False.",
        )
    if receiver.arg_type is None:
        return NotCovered()
    if not is_tensor_type(receiver.arg_type):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "receiver type is outside the float32 CPU tensor surface",
            "Annotate the classification logits as rextio_torch.types.TensorF32Cpu2D.",
        )
    if receiver.arg_type != TENSOR_F32_CPU_2D:
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f"{method} requires float32 CPU rank-2 logits; got {receiver.arg_type!r}",
            "Use TensorF32Cpu2D for the classification-head receiver.",
        )
    return Claimed(rule_id=rule_id, result_type=result_type)


__all__ = ["ARGMAX_RULE", "CLASSIFICATION_RULES", "SOFTMAX_RULE", "try_claim"]
