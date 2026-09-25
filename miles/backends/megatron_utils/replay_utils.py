from megatron.core.utils import unwrap_model

from miles.utils.replay_base import Replay


def get_replay_columns(kind: str, models) -> list[tuple[Replay, int]]:
    is_value = kind == "value"
    dense_prefix = unwrap_model(models[0]).config.mova_num_dense_layers if is_value else 0

    columns = []
    for model in unwrap_model(models):
        for layer in model.decoder.layers:
            owner = getattr(layer.self_attention, "value_projection", None) if is_value else layer.mlp
            replay = getattr(getattr(owner, "router", None), "routing_replay", None)
            if replay is not None:
                columns.append((replay, layer.layer_number - 1 - dense_prefix))
    return columns
