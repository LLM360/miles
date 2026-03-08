---

## paths:
  - "miles/utils/ft/**/*.py"

## ft package architecture rules

### No Ray/K8s outside adapters

Code outside `adapters/` (formerly `platform/`) must NOT reference Ray or Kubernetes:

- No `import ray`, `from ray`, or `.remote()` calls
- No `import kubernetes` or `from kubernetes`
- Exception: `fault_injectors/` (test-only tooling, Ray usage allowed)
- Any Ray/K8s interaction must live in `adapters/` and be exposed to other layers via Protocol interfaces

### Module dependency rules

Layers: `agents/`, `controller/`, `adapters/`, `factories/`, `cli/`, `utils/`

Allowed dependencies (`A → B` means A may import from B):


| Module                         | May import from                                                         | Must NOT import from                                                         |
| ------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `agents/` (excl. types.py)     | `agents/types.py`, `adapters/types.py`, `utils/`                        | `controller/`, `adapters/impl/`, `factories/`, `cli/`                        |
| `controller/` (excl. types.py) | `controller/types.py`, `agents/types.py`, `adapters/types.py`, `utils/` | `agents/` (excl. types.py), `adapters/impl/`, `factories/`, `cli/`           |
| `adapters/impl/`               | `adapters/types.py`, `utils/`                                           | `controller/`, `agents/`, `factories/`, `cli/`                               |
| `adapters/stubs.py`            | `adapters/types.py`, `utils/`                                           | `controller/`, `agents/`, `factories/`                                       |
| `factories/`                   | everything (composition root)                                           | —                                                                            |
| `cli/`                         | `factories/`, `adapters/types.py`, `controller/types.py`, `utils/`      | `adapters/impl/`, `agents/` (excl. types.py), `controller/` (excl. types.py) |


Key principles:

- `factories/` is the composition root — the ONLY place allowed to import across all layers
- `adapters/impl/` must NOT import from `factories/`; use builder injection instead (factory passes builder fn to adapter)
- `controller/types.py` may import from `agents/types.py` (one-way; DiagnosticResult etc.)
- All `types.py` files may import from `utils/base_model.py`
- No circular dependencies between `types.py` files

## Error-as-Empty — FORBIDDEN on safety-critical paths

On fault detection / health check / recovery / diagnostic paths, "I don't know" must never look like "everything is fine."

- Do NOT catch exceptions and return empty (`[]`, `None`, `set()`, `EMPTY_DF`) when callers interpret empty as "all clear"
- Let exceptions propagate, or use a result type that distinguishes success-empty from error

