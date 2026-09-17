"""Dynamic batching must preserve token estimates and VPP scheduling groups."""

from argparse import Namespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.training_utils import data as data_module
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


def _iterators(monkeypatch, *, sample_count, token_budget, vpp_size, group_size):
    group = GroupInfo(rank=0, size=1, group=None)
    state = ParallelState(
        intra_dp=group,
        intra_dp_cp=group,
        cp=group,
        tp=group,
        vpp_size=vpp_size,
        microbatch_group_size_per_vp_stage=group_size,
    )
    monkeypatch.setattr(data_module, "get_parallel_state", lambda: state)
    # Exercise the real count calculation, tensor rounding, partitioning and
    # iterators on CPU; only device selection and distributed communication
    # are replaced. Two rollout steps also exercise partition index offsets.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", Mock())
    args = Namespace(
        use_dynamic_global_batch_size=False,
        use_dynamic_batch_size=True,
        global_batch_size=sample_count,
        max_tokens_per_gpu=token_budget,
    )
    rollout = {"total_lengths": [4] * (2 * sample_count), "sample_ids": list(range(2 * sample_count))}
    return data_module.get_data_iterator(args, [], rollout)


@pytest.mark.parametrize(
    "sample_count,token_budget,group_size,minimum,expected",
    [
        (8, 32, 2, 1, 2),
        (8, 16, 2, 2, 2),
        (8, 12, 2, 3, 4),
        (8, 8, 2, 4, 4),
        (10, 8, 2, 5, 6),
        (8, 32, 4, 1, 4),
        (8, 12, 4, 3, 4),
    ],
)
def test_vpp_rounds_up_without_losing_samples_or_token_budget(
    monkeypatch, sample_count, token_budget, group_size, minimum, expected
):
    assert data_module.get_minimum_num_micro_batch_size([4] * sample_count, token_budget) == minimum
    iterators, counts = _iterators(
        monkeypatch, sample_count=sample_count, token_budget=token_budget, vpp_size=2, group_size=group_size
    )
    assert counts == [expected, expected]
    assert expected >= minimum and expected % group_size == 0
    assert len(iterators) == 2
    for iterator in iterators:
        seen = []
        for step, count in enumerate(counts):
            for _ in range(count):
                batch = iterator.get_next(["total_lengths", "sample_ids"])
                assert batch["sample_ids"]
                assert sum(batch["total_lengths"]) <= token_budget
                assert all(step * sample_count <= i < (step + 1) * sample_count for i in batch["sample_ids"])
                seen.extend(batch["sample_ids"])
        assert sorted(seen) == list(range(2 * sample_count))


@pytest.mark.parametrize("token_budget,expected", [(32, 1), (12, 3), (8, 4)])
def test_non_vpp_keeps_unrounded_counts(monkeypatch, token_budget, expected):
    iterators, counts = _iterators(
        monkeypatch, sample_count=8, token_budget=token_budget, vpp_size=1, group_size=None
    )
    assert len(iterators) == 1
    assert counts == [expected, expected]


@pytest.mark.parametrize("sample_count,token_budget", [(3, 4), (1, 32)])
def test_vpp_rejects_more_microbatches_than_local_samples(monkeypatch, sample_count, token_budget):
    with pytest.raises(ValueError, match="each DP rank has only.*samples"):
        _iterators(
            monkeypatch, sample_count=sample_count, token_budget=token_budget, vpp_size=2, group_size=2
        )
