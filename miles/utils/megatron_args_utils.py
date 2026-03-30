"""
Utils for megatron arguments, but not related to megatron core logic
"""


def compute_megatron_dp_size(args):
    """src: Megatron arguments.py :: validate_args"""
    total_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    return args.world_size // total_model_size
