# Fault Tolerance E2E Tests

## Test Overview

| Test | Type | What it verifies |
|------|------|-----------------|
| `test_trainer_ft_no_failure.py` | Comparison | indep_dp matches normal DP when no faults |
| `test_trainer_ft_with_failure.py` | Comparison, multi-phase | indep_dp matches normal DP after fault + ckpt resume |
| `test_trainer_ft_deterministic.py` | Comparison | indep_dp matches normal DP with stop+start healing (no missed steps) |
| `test_ft_random.py` | Non-comparison | System survives random crashes without hanging |

## Mode Variants

Each test runs with `--mode`:

All modes are **disaggregated** (training and rollout on separate nodes). Modes without rollout use debug rollout data.

| Mode | Nodes | DP cells | Batch | Parallelism | Rollout | Coverage |
|------|-------|----------|-------|-------------|---------|----------|
| `dp2_cp2_tp2_ep2` | 1 | 2 | 3 | CP2 TP2 EP2 | debug data | TP + EP |
| `dp2_cp2_pp2` | 1 | 2 | 3 | CP2 PP2 | debug data | PP |
| `dp4_cp2` | 1 | 4 | 5 | CP2 | debug data | Multi-replica (>=4 cells) |
| `dp2_cp2_real_rollout` | 1 | 2 | 3 | CP2 | 4 engines × 1 GPU | Real weight update path |
| `6node_dp4_cp2_tp2_pp2_ep2_etp2` | 4+2 | 4 | 5 | CP2 TP2 PP2 EP2 ETP2 | 2 engines × 8 GPU | Large-scale, all parallelism |

Batch sizes are deliberately **not** divisible by num_cells to test uneven sample distribution across replicas (e.g. DP4 + batch 5 → 2,1,1,1).

## Running

### Comparison tests (`test_trainer_ft_no_failure.py`, `test_trainer_ft_with_failure.py`, `test_trainer_ft_deterministic.py`)

These compare baseline (normal DP) against target (indep_dp). They support 4 subcommands:

- `run` — full pipeline: prepare + baseline + target + compare
- `baseline` / `target` — run one side independently (useful for debugging)
- `compare` — re-run comparison on existing dumps (no GPU needed)

```bash
# Full pipeline (CI)
python tests/e2e/ft/test_trainer_ft_no_failure.py run --mode dp2_cp2_tp2_ep2

# Step by step (debugging)
python tests/e2e/ft/test_trainer_ft_no_failure.py baseline --mode dp2_cp2_tp2_ep2 --dump-dir /tmp/ft
python tests/e2e/ft/test_trainer_ft_no_failure.py target   --mode dp2_cp2_tp2_ep2 --dump-dir /tmp/ft
python tests/e2e/ft/test_trainer_ft_no_failure.py compare  --mode dp2_cp2_tp2_ep2 --dump-dir /tmp/ft

# With-failure has phases (phase_a saves ckpt, phase_b resumes + injects fault)
python tests/e2e/ft/test_trainer_ft_with_failure.py run --mode dp4_cp2

# Deterministic healing (designed for large-scale disagg)
python tests/e2e/ft/test_trainer_ft_deterministic.py run --mode 6node_dp4_cp2_tp2_pp2_ep2_etp2
```

### Non-comparison tests (`test_ft_random.py`)

Single `run` subcommand. No baseline — just verifies the system doesn't crash.

```bash
python tests/e2e/ft/test_ft_random.py run --mode dp4_cp2 --seed 42 --num-steps 50
```

## Debug Rollout Data

FT tests use pre-recorded rollout data (`--load-debug-rollout-data --debug-train-only`) to skip real rollout generation and save GPU resources.

`conftest_ft/execution.py` `prepare()` downloads the data via `U.hf_download_dataset()`. Then `get_common_train_args()` passes `--load-debug-rollout-data` and `--debug-train-only` to the training command (for modes with `rollout_gpus == 0`).

### How to regenerate

Any `run_*.py` script with `--mode debug_minimal` already dumps rollout data via `--save-debug-rollout-data` (part of dump details):

```bash
# Step 1: Run with debug_minimal to dump rollout data
python scripts/run_qwen3_30b_a3b.py --mode debug_minimal --num-rollout 10

# Step 2: Locate the dumped rollout data
ls /root/output/<run_id>/dump_details/

# Step 3: Upload to HF
huggingface-cli upload --repo-type dataset fzyzcjy/miles-ft-test-debug-rollout-data <path>
```
