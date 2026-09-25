"""Unit tests for rollout-routing-replay column mapping."""

from types import SimpleNamespace

from miles.backends.megatron_utils.replay_utils import get_replay_columns
from miles.utils.replay_base import RoutingReplayManager


def _router(manager: RoutingReplayManager) -> SimpleNamespace:
    """A router carrying the replay registered for it, like TopKRouter.__init__ leaves it."""
    replay = manager.create_replay()
    return SimpleNamespace(routing_replay=replay)


def _chunk(manager: RoutingReplayManager, layer_numbers, num_dense_layers: int) -> SimpleNamespace:
    """A model chunk whose layers above the dense prefix carry both routers.

    Layers are built back to front, so the mapping must follow the layer number, not the
    build order.
    """
    layers = []
    for layer_number in sorted(layer_numbers, reverse=True):
        layer = SimpleNamespace(layer_number=layer_number, mlp=SimpleNamespace(), self_attention=SimpleNamespace())
        if layer_number > num_dense_layers:
            layer.mlp.router = _router(manager)
            layer.self_attention.value_projection = SimpleNamespace(router=_router(manager))
        layers.append(layer)
    layers.reverse()
    config = SimpleNamespace(mova_num_dense_layers=num_dense_layers)
    return SimpleNamespace(config=config, decoder=SimpleNamespace(layers=layers))


def _columns(pairs):
    return [column for _, column in pairs]


def _replays(pairs):
    return [replay for replay, _ in pairs]


def test_columns_are_global_layer_indices():
    """Value routing is captured for MoVA layers only, so its columns start after the dense prefix."""
    manager = RoutingReplayManager()
    chunks = [_chunk(manager, [1, 2, 3], 2), _chunk(manager, [4, 5, 6], 2)]

    assert _columns(get_replay_columns("ffn", chunks)) == [2, 3, 4, 5]
    assert _columns(get_replay_columns("value", chunks)) == [0, 1, 2, 3]


def test_pairing_follows_layer_not_registration_order():
    manager = RoutingReplayManager()
    chunk = _chunk(manager, range(1, 7), 2)

    assert _replays(get_replay_columns("ffn", [chunk])) == [
        layer.mlp.router.routing_replay for layer in chunk.decoder.layers[2:]
    ]
    assert _replays(get_replay_columns("value", [chunk])) == [
        layer.self_attention.value_projection.router.routing_replay for layer in chunk.decoder.layers[2:]
    ]


def test_ffn_mapping_needs_no_mova_config():
    """Vanilla MoE: no dense prefix, no MoVA config fields, no value projection anywhere."""
    manager = RoutingReplayManager()
    chunk = _chunk(manager, range(1, 7), num_dense_layers=0)
    chunk.config = SimpleNamespace()

    assert _columns(get_replay_columns("ffn", [chunk])) == [0, 1, 2, 3, 4, 5]
