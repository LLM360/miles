from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.replay_base import BaseReplayManager, RoutingReplayManager


def get_replay_layer_indices(models) -> list[int]:
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)
    return layer_indices


def _register_replay_list_moe(
    replay_list,
    replay_data,
    models,
    *,
    source_layer_indices=None,
):
    layer_indices = get_replay_layer_indices(models)
    if source_layer_indices is None:
        replay_columns = layer_indices
    else:
        source_layer_indices = [int(layer_idx) for layer_idx in source_layer_indices]
        if source_layer_indices != layer_indices:
            raise ValueError(
                f"routing replay layer shard {source_layer_indices} does not match local model layers {layer_indices}"
            )
        replay_columns = range(len(layer_indices))

    for replay_idx, replay_column in enumerate(replay_columns):
        layer_data = replay_data[:, replay_column]
        replay_list[replay_idx].record(layer_data)


def get_register_replay_list_func(manager: BaseReplayManager):
    if isinstance(manager, RoutingReplayManager):
        return _register_replay_list_moe
    else:
        raise ValueError(f"Unsupported manager type: {type(manager)}")
