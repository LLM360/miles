from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.replay_base import BaseReplayManager, IndexerReplayManager, RoutingReplayManager


class _FakeReplay:
    def __init__(self, *top_indices):
        self.top_indices = list(top_indices)

    def pop_forward(self):
        return self.top_indices.pop(0)

    def pop_backward(self):
        return self.pop_forward()


def _topk(scores, topk):
    return torch.topk(scores, topk, dim=1).indices.to(torch.int32)


def _make_replay_manager(top_indices, manager_type=BaseReplayManager):
    manager = manager_type()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


@pytest.mark.parametrize("manager_type", [BaseReplayManager, IndexerReplayManager])
def test_get_topk_fn_fills_all_invalid_rows_with_arange(manager_type):
    # an all-(-1) row is a masked/pad token; fill it with arange to avoid reading
    # invalid (-1) positions downstream
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    manager = _make_replay_manager(torch.tensor([[-1, -1, -1]], dtype=torch.int32), manager_type)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), torch.tensor([[0, 1, 2]], dtype=torch.int32))


@pytest.mark.parametrize("manager_type", [BaseReplayManager, IndexerReplayManager])
def test_get_topk_fn_preserves_partial_padding(manager_type):
    # a row with some valid picks keeps its -1 padding (only all-(-1) rows are filled)
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    replayed_top_indices = torch.tensor([[2, -1, -1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed_top_indices, manager_type)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), replayed_top_indices)


@pytest.mark.parametrize("stage", ["replay_forward", "replay_backward"])
@pytest.mark.parametrize("return_probs", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_routing_replay_fills_missing_experts_without_duplicates(stage, return_probs, dtype):
    scores = torch.arange(5, dtype=torch.float32).repeat(3, 1)
    replayed = torch.tensor([[2, -1, -1], [-1, -1, -1], [4, 1, 2]], dtype=dtype)
    original = replayed.clone()
    manager = _make_replay_manager(replayed, RoutingReplayManager)
    manager.stage = stage

    result = manager.get_topk_fn(_topk, return_probs)(scores, 3)

    expected = torch.tensor([[2, 4, 3], [4, 3, 2], [4, 1, 2]], dtype=dtype)
    if return_probs:
        probs, indices = result
        torch.testing.assert_close(probs, scores.gather(1, expected.long()))
    else:
        indices = result
    torch.testing.assert_close(indices, expected)
    torch.testing.assert_close(replayed, original)
