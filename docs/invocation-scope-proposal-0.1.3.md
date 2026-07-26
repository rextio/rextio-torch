# Core capability proposal: one RAII no-grad scope per generated native invocation

**Status:** proposal-only for the rextio-torch **0.1.3 candidate**

**Does not modify Core** (`../rextio` is read-only for this work)

**Does not activate a production optimization**

## Read-only Core API 1.6 inspection (evidence)

Live Core host (`rextio` plugin API **1.6**, published core **0.1.6**):

| Surface | Observation |
| --- | --- |
| `LoweredExpr` fields | Only `rust`, `uses`, `helpers` |
| `RextioLoweringPlugin` hooks | `type_vocabulary`, `claim`, `lower`, `crate_dependencies` (+ v2 `covers`/`describe`) |
| Function body prelude / epilogue | **Absent** |
| Per-generated-function / per-invocation scope hook | **Absent** |

Module-level `helpers` and `uses` are shared across the generated crate module.
They are **not** a deterministic stack-scoped prelude for one native call.
Core owns statements, control flow, and function-body emission; plugins may
emit only a single expression per claim site plus module support items
([plugin-lowering.md §3](https://github.com/rextio/rextio/blob/0.1.6/docs/specs/plugin-lowering.md)
 on the released Core **0.1.6** tag).

Repository probe: `rextio_torch.invocation_scope.inspect_core_function_scope_capability()`.

## Production safety baseline (active)

Every Alpha tch helper in `rextio_torch.rust_snippets.ops` installs:

```rust
let _guard = tch::no_grad_guard();
```

inside the helper body. Properties this package relies on:

1. **Stack RAII** — Drop restores prior grad mode on every exit, including
   `?`-mapped `TchError` paths.
2. **No shared mutable state** — no process globals, thread locals, or
   tensor-lifetime tricks for grad mode.
3. **Reentrancy-safe relative to helper boundaries** — nested helper calls
   nest guards; Python callbacks that re-enter other helpers do not inherit a
   leaked outer “module flag”.
4. **Identical exception semantics** for the production path: a failure in one
   op cannot leave a process-wide no-grad latch stuck.

`INVOCATION_SCOPE_OPTIMIZATION_ACTIVE` remains **`False`**.

## Why a plugin-local “single guard” is not implemented without Core

A single `no_grad_guard` around an entire generated function body would avoid
redundant guard enter/leave on multi-op pipelines (for example small-batch
scoring). Implementing that **without** a Core body hook would require one of:

- wrapping every expression in an artificial block that cannot cover
  intervening Core-owned control flow,
- stashing grad mode in globals / thread locals,
- or tying grad mode to tensor lifetimes,

all of which can leak across Python callbacks, exceptions, reentrancy, or
nested generated calls. Those approaches are **rejected** for this candidate.

## Proposed Core extension (illustrative)

Target framing: plugin API **1.7+** / a future Core that owns body emission.

```text
class RextioFunctionScopePlugin(Protocol):  # optional, all-or-none with docs
    def function_body_support(
        self,
        function: GeneratedFunctionMeta,  # qualname, claimed rule ids, types
        ctx: FunctionScopeContext,        # backend, fresh_name, language
    ) -> FunctionBodySupport | None:
        ...
```

```text
@dataclass(frozen=True)
class FunctionBodySupport:
    rust_prelude: tuple[str, ...]   # injected once after param binding
    rust_epilogue: tuple[str, ...]  # optional; prefer RAII Drop
    uses: tuple[str, ...] = ()
    helpers: tuple[str, ...] = ()
```

### Required Core guarantees

1. Prelude runs **once per native invocation** of that generated function,
   after parameters are bound, before the first body statement.
2. Prelude values are ordinary Rust locals (RAII). Drop order is reverse
   construction on **every** exit path (return, `?`, unwind).
3. Nested native→native calls (if ever allowed for plugin-typed graphs) nest
   scopes; there is no process-global latch.
4. Core never invents grad-mode state on behalf of plugins; only the plugin’s
   prelude text runs.
5. Emission is deterministic: identical function + claims → identical prelude
   text, deduplicated `uses`/`helpers` as today.
6. Absence of the optional hook (or API &lt; 1.7) preserves today’s behavior.

### rextio-torch intended use (after Core lands)

```rust
// once, at the start of each generated native function body
let _rxttorch_invocation_no_grad = tch::no_grad_guard();
```

Per-operation helpers could then drop their inner guards **only if** a
reviewed, tested policy proves identical exception and reentrancy semantics.
Until then, helpers retain inner guards even if a body prelude also exists
(redundant but safe). The 0.1.3 candidate does **not** remove per-op guards.

### Explicit non-goals of this proposal

- Training / autograd correctness inside the scope (Alpha remains inference).
- CUDA semantics changes.
- Thread-local grad mode owned by the plugin.
- Any claim that 0.1.3 already emits a single invocation-scope guard.

## Tests that pin the limitation

| Test | Asserts |
| --- | --- |
| `tests/test_invocation_scope_proposal.py` | Core fields/hooks, flag false, proposal payload shape |
| Production lower/unit helpers | `no_grad_guard` still present per op |
| Small-batch diagnostic path | Timings labeled diagnostic; no speedup claim |

## Residual limitations

- Multi-op native pipelines pay one tch no-grad enter/leave per helper call.
- Until Core lands a body-scope hook, no strictly plugin-local optimization
  with proven-identical safety is available.
- This document is not a Core commitment; implementers must re-inspect the
  live host API before enabling any optimization.
