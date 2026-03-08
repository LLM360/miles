---
paths:
  - "miles/utils/ft/**/*.py"
---

## ft package architecture rule

Code outside `platform/` must NOT reference Ray or Kubernetes:
- No `import ray`, `from ray`, or `.remote()` calls
- No `import kubernetes` or `from kubernetes`
- Exception: `fault_injectors/` (test-only tooling, Ray usage allowed)
- Any Ray/K8s interaction must live in `platform/` and be exposed to other layers via Protocol interfaces

## Error-as-Empty — FORBIDDEN on safety-critical paths

On fault detection / health check / recovery / diagnostic paths, "I don't know" must never look like "everything is fine."

- Do NOT catch exceptions and return empty (`[]`, `None`, `set()`, `EMPTY_DF`) when callers interpret empty as "all clear"
- Let exceptions propagate, or use a result type that distinguishes success-empty from error
