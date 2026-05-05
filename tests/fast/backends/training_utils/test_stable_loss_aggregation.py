"""Numerical contracts for the stable loss modes after the upstream port."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import loss as loss_utils


@pytest.fixture
def parallel_state(monkeypatch):
    state = SimpleNamespace(
        cp=SimpleNamespace(size=1),
        intra_dp=SimpleNamespace(size=2),
        intra_dp_cp=SimpleNamespace(size=2),
        is_ulysses_cp=False,
    )
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: state)
    return state


@pytest.mark.parametrize(
    "mode,legacy_flag,megatron,expected,gradient,normalizer",
    [
        (None, False, False, 6.0, [0.25, 0.25, 0.5], 1),
        ("sample-mean", True, False, 6.0, [0.25, 0.25, 0.5], 1),
        ("token-mean", False, False, 32 / 3, [2 / 3] * 3, 3),
        ("token-sum", False, False, 16.0, [1.0] * 3, 3),
        (None, True, False, 16.0, [1.0] * 3, 3),
        ("sample-mean", True, True, 12.0, [0.5, 0.5, 1.0], 1),
        ("token-mean", False, True, 16.0, [1.0] * 3, 3),
        ("token-sum", False, True, 16.0, [1.0] * 3, 3),
    ],
)
def test_mode_values_and_gradients(
    parallel_state, monkeypatch, mode, legacy_flag, megatron, expected, gradient, normalizer
):
    args = Namespace(
        calculate_per_token_loss=legacy_flag,
        loss_agg_mode=mode,
        qkv_format="thd",
        recompute_loss_function=False,
        use_dynamic_global_batch_size=False,
        global_batch_size=4,
        true_on_policy_mode=False,
        allgather_cp=False,
    )
    batch = {"total_lengths": [3, 2], "response_lengths": [2, 1], "loss_masks": [torch.ones(2), torch.ones(1)]}
    logits = torch.tensor([2.0, 6.0, 8.0], requires_grad=True)
    monkeypatch.setattr(loss_utils, "get_loss_function", lambda args: lambda args, batch, x, reduce: (reduce(x), {}))
    value, count, _ = loss_utils.loss_function(args, batch, 2, logits, apply_megatron_loss_scaling=megatron)
    assert value.item() == pytest.approx(expected)
    assert count.item() == normalizer
    value.backward()
    torch.testing.assert_close(logits.grad, torch.tensor(gradient))


def test_rollout_denominators_remain_supported(parallel_state):
    reducer = cp_utils.get_sum_of_sample_mean(
        [3, 2],
        [2, 1],
        [torch.ones(2), torch.ones(1)],
        denominators=[torch.tensor(4), torch.tensor(4)],
        loss_agg_mode="sample-mean",
    )
    assert reducer(torch.tensor([2.0, 6.0, 8.0])).item() == 4.0


@pytest.mark.parametrize("mode", ["sample-mean", "token-mean", "token-sum"])
def test_fully_masked_samples_stay_finite(parallel_state, mode):
    reducer = cp_utils.get_sum_of_sample_mean([3], [2], [torch.zeros(2)], loss_agg_mode=mode)
    x = torch.tensor([2.0, 6.0], requires_grad=True)
    value = reducer(x)
    assert value.item() == 0.0
    value.backward()
    torch.testing.assert_close(x.grad, torch.zeros(2))


@pytest.mark.parametrize("mode,expected", [("sample-mean", 16 / 3), ("token-mean", 16.0), ("token-sum", 16.0)])
def test_cp_partitions_reconstruct_the_same_reduction(parallel_state, mode, expected):
    # For total length 8 / response length 4, zigzag CP splits the last
    # response token onto rank 0 and the first three onto rank 1.
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    parallel_state.cp.size = 2
    values = []
    for rank, local_x in [(0, [8.0]), (1, [2.0, 100.0, 6.0])]:
        parallel_state.cp.rank = rank
        reducer = cp_utils.get_sum_of_sample_mean([8], [4], [mask], loss_agg_mode=mode)
        values.append(reducer(torch.tensor(local_x)))
    assert sum(values).item() == pytest.approx(expected)
