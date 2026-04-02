MODEL_NAME: str = "Qwen3-30B-A3B-5layer"
MODEL_HF_REPO: str = f"fzyzcjy/{MODEL_NAME}"
MODEL_TYPE: str = "qwen3-30B-A3B"
DEBUG_ROLLOUT_DATA_HF_REPO: str = "fzyzcjy/miles-ft-test-debug-rollout-data"


@dataclass(frozen=True)
class FTTestMode:
    name: str
    model_name: str
    megatron_model_type: str
    num_gpus_total: int
    num_cells: int
    parallel_args: str
    rollout_gpus: int
    num_steps: int = 10


MODES: dict[str, FTTestMode] = {
    "dp2_cp2_tp2_ep2": FTTestMode(
        name="dp2_cp2_tp2_ep2",
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_total=8,
        num_cells=2,
        rollout_gpus=0,
        parallel_args=(
            "--tensor-model-parallel-size 2 --context-parallel-size 2 "
            "--expert-model-parallel-size 2 --sequence-parallel"
        ),
    ),
    "dp2_cp2_pp2": FTTestMode(
        name="dp2_cp2_pp2",
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_total=8,
        num_cells=2,
        rollout_gpus=0,
        parallel_args="--pipeline-model-parallel-size 2 --context-parallel-size 2",
    ),
    "dp4_cp2": FTTestMode(
        name="dp4_cp2",
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_total=8,
        num_cells=4,
        rollout_gpus=0,
        parallel_args="--context-parallel-size 2",
    ),
    "dp2_cp2_real_rollout": FTTestMode(
        name="dp2_cp2_real_rollout",
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_total=8,
        num_cells=2,
        rollout_gpus=4,
        parallel_args="--context-parallel-size 2",
    ),
}


def resolve_mode(mode: str) -> FTTestMode:
    if mode not in MODES:
        raise typer.BadParameter(f"Unknown mode {mode!r}, valid: {list(MODES.keys())}")
    return MODES[mode]
