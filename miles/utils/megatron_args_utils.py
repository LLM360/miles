"""
Utils for megatron arguments, but not related to megatron core logic
"""


def compute_megatron_world_size_per_replica(args) -> int:
    return args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size


def compute_megatron_dp_size(args, total_gpus: int) -> int:
    """src: Megatron arguments.py :: validate_args"""
    return total_gpus // compute_megatron_world_size_per_replica(args)
