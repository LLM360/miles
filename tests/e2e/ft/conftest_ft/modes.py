# NOTE: Please refer to tests/e2e/ft/README.md for documentations and source-of-truth

from dataclasses import dataclass

import typer

MODEL_NAME: str = "Qwen3-30B-A3B-5layer"
MODEL_HF_REPO: str = f"fzyzcjy/{MODEL_NAME}"
MODEL_TYPE: str = "qwen3-30B-A3B"
DEBUG_ROLLOUT_DATA_HF_REPO: str = "fzyzcjy/miles-ft-test-debug-rollout-data"


@dataclass(frozen=True)
class FTTestMode:
    model_name: str
    megatron_model_type: str
    num_cells: int
    parallel_args: str
    train_num_nodes: int = 1
    train_gpus_per_node: int = 8
    rollout_num_engines: int = 0
    rollout_gpus_per_engine: int = 0
    num_steps: int = 10
    global_batch_size: int = 5

    @property
    def has_rollout(self) -> bool:
        return self.rollout_num_engines > 0

    @property
    def total_rollout_gpus(self) -> int:
        return self.rollout_num_engines * self.rollout_gpus_per_engine


MODES: dict[str, FTTestMode] = {
    # --- 1-node (8 GPUs) variants ---
    "dp2_cp2_tp2_ep2": FTTestMode(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        global_batch_size=3,
        parallel_args=(
            "--tensor-model-parallel-size 2 "
            "--context-parallel-size 2 "
            "--expert-model-parallel-size 2 "
            "--sequence-parallel"
        ),
    ),
    "dp2_cp2_pp2": FTTestMode(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        global_batch_size=3,
        parallel_args=(
            "--pipeline-model-parallel-size 2 "
            "--context-parallel-size 2"
        ),
    ),
    "dp4_cp2": FTTestMode(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_cells=4,
        parallel_args="--context-parallel-size 2",
    ),
    "dp2_cp2_real_rollout": FTTestMode(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        global_batch_size=3,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        parallel_args="--context-parallel-size 2",
    ),
    # --- 6-node (48 GPUs) disaggregated: 4 train nodes + 2 rollout nodes ---
    "6node_dp4_cp2_tp2_pp2_ep2_etp2": FTTestMode(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_cells=4,
        train_num_nodes=4,
        train_gpus_per_node=8,
        rollout_num_engines=2,
        rollout_gpus_per_engine=8,
        parallel_args=(
            "--tensor-model-parallel-size 2 "
            "--context-parallel-size 2 "
            "--pipeline-model-parallel-size 2 "
            "--expert-model-parallel-size 2 "
            "--expert-tensor-parallel-size 2 "
            "--sequence-parallel"
        ),
    ),
}


def resolve_mode(mode: str) -> FTTestMode:
    if mode not in MODES:
        raise typer.BadParameter(f"Unknown mode {mode!r}, valid: {list(MODES.keys())}")
    return MODES[mode]
