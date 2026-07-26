# One RAII no-grad scope per eligible native invocation

**Status:** implemented-candidate for rextio-torch **0.1.3**

**Core contract:** Rextio **0.1.7 candidate**, plugin API **1.7**

## Active contract

`RextioTorchPlugin.function_scope_guard(ctx)` returns:

```python
PluginFunctionScopeGuard(rust="tch::no_grad_guard()")
```

only when all of the following are true:

1. the backend is the PyO3 host extension,
2. the function contains at least one rextio-torch operation claim, and
3. the function contains no in-process Python fallback call (RXT075).

Core owns the binding name. It converts materialized tensor inputs first,
installs the guard, executes the native body, evaluates the native result,
drops the guard, and only then converts the result back to `torch.Tensor`.
Rust RAII restores the caller's prior grad mode on early returns and
`?`/`TchError` error exits.

## Two helper families

Core sets `LoweringContext.function_scope_guard_active=True` only for the
provider whose guard it actually emitted. Every Torch lowerer uses that exact
boolean to select one of two distinct Rust symbols:

| Context | Example helper | Guard ownership |
| --- | --- | --- |
| inactive / legacy / RXT075 | `__rxttorch_relu` | helper contains `let _guard = tch::no_grad_guard();` |
| active API 1.7 function | `__rxttorch_relu_function_scoped` | enclosing generated function owns the guard |

Distinct names allow active and inactive functions to coexist in one generated
Rust module without helper deduplication or duplicate-definition ambiguity.
Missing legacy context attributes and unexpected truthy values fail closed to
the guarded helper family.

The scope-aware selection covers the complete current lower surface: linear,
ReLU/sigmoid/tanh, bounded unary math, GELU, mean/sum, add/mul/sub/div, matmul,
softmax/argmax, and the build-only CUDA operation slice.

## RXT075 and type-only safety

A function that calls Python fallback must not hold a whole-function no-grad
guard across that callback. The hook returns `None` when
`ctx.has_python_boundary_calls` is true, so every operation uses its local
per-operation guard and the callback observes the ambient Python grad mode.

Core can also call a plugin hook when only a namespaced plugin type appears in
a signature. rextio-torch returns `None` when `ctx.used_rule_ids` is empty;
type conversion alone does not pay a no-grad scope or activate guardless
helpers.

Standalone backends currently return `None`. The Torch Alpha surface remains a
PyO3/tch host-extension integration.

## Safety properties

- Stack RAII only; no process globals, thread locals, or tensor-lifetime flags.
- Native outputs remain inference/no-grad (`requires_grad=False`).
- Early return and `TchError` paths restore the previous grad mode.
- RXT075 callbacks run outside any Torch whole-function guard.
- API 1.1–1.6/no-hook contexts retain the original guarded helper behavior.

## Evidence

| Test | Contract |
| --- | --- |
| `tests/test_invocation_scope_proposal.py` | hook eligibility, type-only/RXT075 decline, distinct helper families, legacy-context fallback |
| `tests/e2e/test_phase_a_real_cargo.py` | one guard per eligible function, active/inactive helper coexistence and compilation, callback ambient grad mode, early return/error restoration, no-grad output |
| Existing lower/unit suites | every claimed Torch operation preserves its prior validation and Rust operation |

The Core 0.1.7 candidate is pinned by exact Git commit in CI until it is
formally released. This candidate work does not publish or tag either project.
