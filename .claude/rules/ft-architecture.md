---
paths:
  - "miles/utils/ft/**/*.py"
---

## ft architecture rules

### No Ray/K8s outside adapters

Code outside `adapters/` must NOT reference Ray or Kubernetes (`import ray`, `.remote()`, `import kubernetes`).
Exception: `fault_injectors/` (test-only).

### Layer dependencies

From top to bottom: `cli` > `factories` > `controller` > `agents` > `adapters` > `utils`

Each layer may only import from layers below it. `types.py` is each layer's public API; only `factories/` (composition root) may import implementation files from other layers.

- `adapters/impl/` receives builder functions via injection, never imports upward
- `controller/types.py` may import `agents/types.py` (one-way; no reverse)
- `utils/` has no types.py — all files are public

### Error-as-Empty — FORBIDDEN on safety-critical paths

On fault detection / recovery / diagnostic paths, "I don't know" must never look like "everything is fine."
Do NOT catch exceptions and return empty (`[]`, `None`, `set()`) when callers interpret empty as "all clear."
