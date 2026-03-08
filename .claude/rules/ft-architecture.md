---
paths:
  - "miles/utils/ft/**/*.py"
---

## ft architecture rules

### No Ray/K8s outside adapters

Code outside `adapters/` must NOT reference Ray or Kubernetes (`import ray`, `.remote()`, `import kubernetes`).
Exception: `fault_injectors/` (test-only).

### Layer dependencies

From top to bottom: `cli` > `factories` > `adapters` > `controller`, `agents` > `utils`

Each layer may only import from layers below it.
Exception: `controller` and `agents` may import `adapters/types.py` (the boundary contract — cross-layer protocols and constants).

- `adapters/types.py` has all cross-layer Protocol definitions; may import from `controller/types.py` and `agents/types.py` (downward)
- `controller/types.py` has data types + controller-internal Protocols; may import from `agents/types.py` (downward)
- `agents/types.py` has data types only; imports from `utils/` only
- `factories/` is the composition root — may import from all layers
- `adapters/impl/` receives builder functions via injection, never imports `controller` or `agents`
- `utils/` has no types.py — all files are public

### Error-as-Empty — FORBIDDEN on safety-critical paths

On fault detection / recovery / diagnostic paths, "I don't know" must never look like "everything is fine."
Do NOT catch exceptions and return empty (`[]`, `None`, `set()`) when callers interpret empty as "all clear."
