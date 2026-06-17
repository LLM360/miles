"""FLOPs balancing also applies when rollout-side scheduling is unavailable."""

from types import SimpleNamespace

import pytest
from tests.fast.utils.test_dp_schedule import make_args

from miles.backends.training_utils import data as training_data
from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw
from miles.utils.flops_utils import calculate_workloads

_LENGTHS = [40, 71, 308, 310, 327, 345, 363, 411]


def _args(enabled):
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=900, balance_data=True, balance_by_flops=enabled)
    args.global_batch_size = 8
    args.qkv_format = "thd"
    args.use_dynamic_global_batch_size = False
    return args


def _peak(parts, args):
    workloads = calculate_workloads(_LENGTHS, args)
    return max(sum(workloads[i] for i in part) for part in parts)


def test_legacy_dp_split_reduces_peak_compute_and_preserves_rows():
    args = _args(True)
    batch = {
        "tokens": [[0] * n for n in _LENGTHS],
        "sample_indices": list(range(8)),
        "domains": ["math", "code"] * 4,
        "all_domains": ["code", "math"],
    }
    flops = split_train_data_by_dp_raw(args, batch, dp_size=2)
    tokens = split_train_data_by_dp_raw(_args(False), batch, dp_size=2)
    assert _peak([shard["partition"] for shard in flops], args) < _peak([shard["partition"] for shard in tokens], args)
    assert sorted(i for shard in flops for i in shard["sample_indices"]) == list(range(8))
    assert [len(shard["tokens"]) for shard in flops] == [4, 4]
    assert all(shard["domains"] == [batch["domains"][i] for i in shard["partition"]] for shard in flops)
    assert all(shard["all_domains"] == ["code", "math"] for shard in flops)


@pytest.mark.parametrize("enabled", [False, True])
def test_training_side_scheduler_keeps_each_sample_once(monkeypatch, enabled):
    parallel = SimpleNamespace(
        effective_dp=SimpleNamespace(size=1, group=None),
        cp=SimpleNamespace(size=1),
        vpp_size=1,
        microbatch_group_size_per_vp_stage=None,
    )
    monkeypatch.setattr(training_data, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(training_data.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(
        training_data.GeneralPGUtil, "create", lambda group: SimpleNamespace(all_reduce=lambda *a, **k: None)
    )
    iterators, counts = training_data.get_data_iterator(_args(enabled), None, {"total_lengths": _LENGTHS})
    parts = iterators[0].micro_batch_indices
    assert sorted(i for part in parts for i in part) == list(range(8))
    assert counts == [3]
    assert len(parts) == 3 and all(parts)
    # With this mixed-length batch the FLOPs objective improves the compute
    # balance while still covering exactly the same training examples.
    if enabled:
        baseline, _ = training_data.get_data_iterator(_args(False), None, {"total_lengths": _LENGTHS})
        assert _peak(parts, _args(True)) <= _peak(baseline[0].micro_batch_indices, _args(True))
