from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.megatron_utils import actor
from miles.ray import rollout
from miles.utils.types import Sample


def _validator_args():
    return SimpleNamespace(
        advantage_estimator="grpo",
        normalize_advantages=False,
        use_rollout_logprobs=False,
        use_critic=False,
        qkv_format="thd",
        compute_advantages_and_returns=True,
        n_samples_per_prompt=8,
        grpo_group_size=2,
        generate_multi_samples=None,
        global_batch_size=32,
        data_parallel_size=8,
        context_parallel_size=1,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=4,
    )


def _rollout_batch(*, mask, removed_by_filter=None):
    response_length = len(mask)
    batch = {
        "rewards": [0.25],
        "response_lengths": [response_length],
        "total_lengths": [response_length + 2],
        "loss_masks": [torch.tensor(mask, dtype=torch.int32)],
        "tokens": [torch.arange(response_length + 2)],
        "truncated": [0],
    }
    if removed_by_filter is not None:
        batch["removed_by_filter"] = [removed_by_filter]
    return batch


def test_validator_allows_only_explicitly_filtered_zero_mask(monkeypatch):
    monkeypatch.setattr(actor, "get_parallel_state", lambda: None)
    logger = MagicMock()

    actor.validate_rollout_for_grpo_training_step(
        _validator_args(),
        _rollout_batch(mask=[0, 0, 0], removed_by_filter=True),
        logger=logger,
    )

    warnings = "\n".join(call.args[0] for call in logger.warning.call_args_list)
    assert "removed_by_filter=True" in warnings


def test_validator_keeps_backward_compatibility_for_active_mask_without_filter_field(monkeypatch):
    monkeypatch.setattr(actor, "get_parallel_state", lambda: None)

    actor.validate_rollout_for_grpo_training_step(
        _validator_args(),
        _rollout_batch(mask=[1, 1, 1]),
        logger=MagicMock(),
    )


@pytest.mark.parametrize(
    "removed_by_filter",
    [None, False],
    ids=["missing-provenance", "explicitly-not-filtered"],
)
def test_validator_rejects_unexplained_zero_mask(monkeypatch, removed_by_filter):
    monkeypatch.setattr(actor, "get_parallel_state", lambda: None)
    logger = MagicMock()

    with pytest.raises(ValueError, match="rollout validation failed"):
        actor.validate_rollout_for_grpo_training_step(
            _validator_args(),
            _rollout_batch(mask=[0, 0, 0], removed_by_filter=removed_by_filter),
            logger=logger,
        )

    errors = "\n".join(call.args[0] for call in logger.error.call_args_list)
    assert "without removed_by_filter=True" in errors


@pytest.mark.parametrize("mask", [[1, 1, 1], [1, -1, 0]], ids=["active", "zero-sum-nonzero"])
def test_validator_rejects_filtered_sample_with_nonzero_mask(monkeypatch, mask):
    monkeypatch.setattr(actor, "get_parallel_state", lambda: None)
    logger = MagicMock()

    with pytest.raises(ValueError, match="rollout validation failed"):
        actor.validate_rollout_for_grpo_training_step(
            _validator_args(),
            _rollout_batch(mask=mask, removed_by_filter=True),
            logger=logger,
        )

    errors = "\n".join(call.args[0] for call in logger.error.call_args_list)
    assert "not all-zero despite removed_by_filter=True" in errors


@pytest.mark.parametrize(
    ("bad_provenance", "expected_error"),
    [
        ("yes", "must be list/tuple"),
        ([], "length mismatch"),
        ([1], "must be bool"),
    ],
    ids=["bad-type", "short-list", "non-bool"],
)
def test_validator_rejects_malformed_filter_provenance(monkeypatch, bad_provenance, expected_error):
    monkeypatch.setattr(actor, "get_parallel_state", lambda: None)
    logger = MagicMock()
    batch = _rollout_batch(mask=[0, 0, 0])
    batch["removed_by_filter"] = bad_provenance

    with pytest.raises(ValueError, match="rollout validation failed"):
        actor.validate_rollout_for_grpo_training_step(
            _validator_args(),
            batch,
            logger=logger,
        )

    errors = "\n".join(call.args[0] for call in logger.error.call_args_list)
    assert expected_error in errors


def test_non_truncated_filter_provenance_reaches_each_dp_shard(monkeypatch):
    manager_class = rollout.RolloutManager.__ray_metadata__.modified_class
    manager = manager_class.__new__(manager_class)
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.args = SimpleNamespace(
        reward_key=None,
        advantage_estimator="ppo",
        rewards_normalization=False,
        use_dynamic_global_batch_size=False,
        balance_by_flops=False,
        balance_data=False,
    )

    filtered = Sample(
        index=0,
        tokens=[10, 11, 12, 13],
        response_length=3,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        remove_sample=True,
    )
    kept = Sample(
        index=1,
        tokens=[20, 21, 22, 23],
        response_length=3,
        reward=1.0,
        status=Sample.Status.COMPLETED,
    )

    train_data = manager._convert_samples_to_train_data([filtered, kept])
    assert train_data["truncated"] == [0, 0]
    assert train_data["removed_by_filter"] == [True, False]
    assert train_data["loss_masks"] == [[0, 0, 0], [1, 1, 1]]

    payloads = []

    class FakeObjectRef:
        def __init__(self, index):
            self.index = index

        def hex(self):
            return f"ref-{self.index}"

    def fake_put(payload):
        payloads.append(payload)
        return FakeObjectRef(len(payloads) - 1)

    monkeypatch.setattr(rollout.ray, "put", fake_put)
    manager._split_train_data_by_dp(train_data, dp_size=2)

    assert [payload["removed_by_filter"] for payload in payloads] == [[True], [False]]
    assert [payload["loss_masks"] for payload in payloads] == [[[0, 0, 0]], [[1, 1, 1]]]
