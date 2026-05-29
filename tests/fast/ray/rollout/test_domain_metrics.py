from types import SimpleNamespace

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.backends.training_utils import data as training_data
from miles.backends.training_utils import log_utils
from miles.ray.rollout import metrics
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    _package_shards,
    convert_samples_to_train_data,
    process_rollout_data_shard,
)
from miles.utils import object_store
from miles.utils.object_store import _field_schemas_for_value


def test_domain_schema_survives_disjoint_shards_and_batch_selection(monkeypatch):
    monkeypatch.setattr(object_store, "FieldSchema", SimpleNamespace)
    args = make_args(rewards_normalization=False, qkv_format="bshd")
    samples = [make_sample(index=i, metadata={"domain": domain}) for i, domain in enumerate(["math", "code", None])]
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    data["total_lengths"] = [len(tokens) for tokens in data["tokens"]]
    shards = _package_shards(args, data, [[0], [1], [2]])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1), pp=SimpleNamespace(size=1))
    monkeypatch.setattr(training_data, "get_parallel_state", lambda: state)
    from miles.backends.training_utils import cp_utils

    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    for shard, expected_domain in zip(shards, ["math", "code", None], strict=True):
        processed = process_rollout_data_shard(args, dict(shard))
        assert processed["domains"] == [expected_domain]
        assert processed["all_domains"] == ["code", "math"]
        schema = _field_schemas_for_value(shard, ROLLOUT_DATA_VALUE_SPEC)
        assert schema["domains"].codec == "msgpack_ragged"
        assert schema["all_domains"].metadata["section"] == "meta_info"
        processed["tokens"] = [torch.tensor(tokens) for tokens in processed["tokens"]]
        processed["max_seq_lens"] = [4]
        processed["loss_masks"] = [torch.tensor(mask) for mask in processed["loss_masks"]]
        iterator = training_data.DataIterator(processed, micro_batch_size=1)
        batch = training_data.get_batch(
            iterator, ["tokens", "max_seq_lens", "loss_masks", "total_lengths", "response_lengths"], qkv_format="bshd"
        )
        assert batch["domains"] == [expected_domain]
        assert batch["all_domains"] == ["code", "math"]


@pytest.mark.parametrize("rewards", [[1, {"score": 2}], [{"score": 2}, 1], [{"score": 2}, {"score": 0}]])
def test_structured_rewards_use_explicit_correctness_and_skip_undefined_means(rewards):
    args = make_args()
    samples = [
        make_sample(index=i, reward=reward, metadata={"domain": "math", "category": "legacy", "correctness_reward": i})
        for i, reward in enumerate(rewards)
    ]
    result = metrics._compute_metrics_from_samples(args, samples)
    assert result["reward/correctness"] == 0.5
    assert result["reward/math/correctness"] == 0.5
    assert "reward/raw_reward" not in result
    assert not any(key.startswith("zero_std/") for key in result)
    assert not any(key.startswith("reward/legacy/") for key in result)
    assert "response_stats/correct/response_len" in result
    assert "response_stats/incorrect/response_len" in result
    assert metrics._compute_passrate_from_samples(make_args(n_samples_per_prompt=2), samples) == {}


def test_explicit_category_and_scalar_correctness_are_preserved():
    args = make_args(log_problem_category="category")
    samples = [
        make_sample(index=i, reward=reward, metadata={"domain": "math", "category": "legacy"})
        for i, reward in enumerate([0, 1])
    ]
    result = metrics._compute_metrics_from_samples(args, samples)
    assert result["reward/legacy/correctness"] == 0.5
    assert result["reward/raw_reward"] == 0.5
    assert "reward/correct/raw_reward" not in result


def test_nominal_rollout_counters_and_reward_mirrors(monkeypatch):
    logged = []
    monkeypatch.setattr(metrics.tracking, "log", lambda args, values, **kwargs: logged.append(values))
    metrics.log_rollout_data(
        2, make_args(rollout_batch_size=3, n_samples_per_prompt=4), [make_sample(reward=1)], {}, 2
    )
    result = logged[0]
    assert result["samples_seen"] == 36  # configured rollouts, not the one surviving training row
    assert result["rollout_step"] == 2
    assert result["reward/correctness"] == result["rollout/reward/correctness"] == 1


def test_domain_names_with_slashes_and_explicit_training_id(monkeypatch):
    logged = []
    monkeypatch.setattr(log_utils.tracking, "log", lambda args, values, **kwargs: logged.append(values))
    result = log_utils.log_train_step(make_args(), {"loss/math/algebra": 2}, 0, 1, 0, 1, train_step=3, should_log=True)
    assert result["optimization/math/algebra/loss"] == 2
    assert result["train/step"] == result["train_step"] == 3
    assert "rollout_step" not in result and "rollout/step" not in result
    assert logged == [result]
