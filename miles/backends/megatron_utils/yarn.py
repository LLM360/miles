"""Keep dense YaRN settings when using Megatron's legacy argument converter."""

from argparse import Namespace


def apply_dense_yarn_config(config, args: Namespace) -> None:
    """Populate the dynamic fields consumed by GPTModel's YaRN embeddings.

    Some Megatron versions only map these fields in argument_utils, while Miles
    uses training.arguments. Preserve values already supplied by a newer converter.
    MLA has separate declared fields and does not use this dense embedding path.
    """
    if getattr(args, "position_embedding_type", None) != "yarn" or getattr(args, "multi_latent_attention", False):
        return
    fields = (
        ("yarn_rotary_scaling_factor", "rotary_scaling_factor", 1.0),
        ("yarn_original_max_position_embeddings", "yarn_original_max_position_embeddings", 4096),
        ("yarn_beta_fast", "yarn_beta_fast", 32.0),
        ("yarn_beta_slow", "yarn_beta_slow", 1.0),
        ("yarn_mscale", "mscale", 1.0),
        ("yarn_mscale_all_dim", "mscale_all_dim", 0.0),
        ("yarn_correction_range_round_to_int", "yarn_correction_range_round_to_int", True),
    )
    for config_name, arg_name, default in fields:
        if not hasattr(config, config_name):
            value = getattr(args, arg_name, None)
            setattr(config, config_name, default if value is None else value)
