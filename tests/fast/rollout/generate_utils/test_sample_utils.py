from unittest.mock import MagicMock

import pytest

from miles.rollout.generate_utils.sample_utils import (
    _merge_sample_pair,
    collect_eval_rewards,
    finalize_eval_rewards,
    persist_then_finalize_eval_rewards,
)
from miles.utils.types import Sample


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode = lambda tokens: f"<decoded:{tokens}>"
    return tokenizer


def make_sample(
    prompt="test_prompt",
    tokens=None,
    response="",
    response_length=0,
    loss_mask=None,
    rollout_log_probs=None,
    status=Sample.Status.COMPLETED,
    label="test_label",
    reward=1.0,
    index=0,
    group_index=0,
    metadata=None,
):
    return Sample(
        prompt=prompt,
        tokens=tokens or [],
        response=response,
        response_length=response_length,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        status=status,
        label=label,
        reward=reward,
        index=index,
        group_index=group_index,
        metadata=metadata or {},
    )


class TestMergeSamples:
    def test_basic_merge(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 3, 10, 11, 12],
            response="response1",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.1, -0.2, -0.3],
        )
        b = make_sample(
            tokens=[1, 2, 3, 10, 11, 12, 20, 21, 30, 31, 32],
            response="response2",
            response_length=3,
            loss_mask=[1, 1, 1],
            rollout_log_probs=[-0.4, -0.5, -0.6],
            status=Sample.Status.TRUNCATED,
        )

        merged = _merge_sample_pair(a, b, mock_tokenizer)

        assert merged.tokens == b.tokens
        assert merged.response_length == 3 + 2 + 3
        assert merged.loss_mask == [1, 1, 1, 0, 0, 1, 1, 1]
        assert merged.rollout_log_probs == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.4, -0.5, -0.6]
        assert merged.prompt == a.prompt
        assert merged.status == b.status
        assert merged.label == a.label
        assert merged.index == a.index
        assert merged.group_index == a.group_index
        assert "response1" in merged.response
        assert "response2" in merged.response
        assert "<decoded:[20, 21]>" in merged.response

    def test_loss_mask_none_defaults_to_all_ones(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=None,
            rollout_log_probs=None,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=None,
            rollout_log_probs=None,
        )

        merged = _merge_sample_pair(a, b, mock_tokenizer)

        assert merged.loss_mask == [1, 0, 1]
        assert merged.rollout_log_probs == [0.0, 0.0, 0.0]

    def test_tokens_prefix_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 3],
            response_length=1,
            loss_mask=[1],
        )
        b = make_sample(
            tokens=[1, 2, 99, 20, 30],
            response_length=1,
            loss_mask=[1],
        )

        with pytest.raises(AssertionError, match="b.tokens must start with a.tokens"):
            _merge_sample_pair(a, b, mock_tokenizer)

    def test_field_mismatch_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
            index=0,
        )
        b = make_sample(
            tokens=[1, 2, 10, 20, 30],
            response_length=1,
            loss_mask=[1],
            index=1,
        )

        with pytest.raises(AssertionError, match="index mismatch"):
            _merge_sample_pair(a, b, mock_tokenizer)

    def test_obs_len_invalid_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10],
            response_length=1,
            loss_mask=[1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 30],
            response_length=1,
            loss_mask=[1],
        )

        with pytest.raises(AssertionError, match="obs_len must be > 0"):
            _merge_sample_pair(a, b, mock_tokenizer)

    def test_sample_validate_fails_raises(self, mock_tokenizer):
        a = make_sample(
            tokens=[1, 2, 10, 11],
            response_length=2,
            loss_mask=[1],
        )
        b = make_sample(
            tokens=[1, 2, 10, 11, 20, 30],
            response_length=1,
            loss_mask=[1],
        )

        with pytest.raises(AssertionError, match="loss_mask length"):
            _merge_sample_pair(a, b, mock_tokenizer)


def test_collect_eval_rewards_selects_named_reward_channel():
    samples = [
        make_sample(reward={"trajectory_reward": 3.0, "success": 1.0}, index=1),
        make_sample(reward={"trajectory_reward": 0.0, "success": 0.0}, index=2),
    ]

    assert collect_eval_rewards(samples, "success") == [1.0, 0.0]


@pytest.mark.parametrize(
    "reward",
    [None, {"trajectory_reward": 2.0}, {"success": float("nan")}],
)
def test_collect_eval_rewards_rejects_invalid_or_missing_channel(reward):
    sample = make_sample(reward=reward, index=17)

    with pytest.raises(RuntimeError, match="refusing to report them as incorrect"):
        collect_eval_rewards([sample], "success")


def test_collect_eval_rewards_reports_bounded_operational_histogram():
    samples = [
        make_sample(
            reward=None,
            index=17,
            status=Sample.Status.ABORTED,
            metadata={
                "multi_attempt": {
                    "invalid_reason": "non_completed_attempt",
                    "attempts": [{"engine_finish_reason": "length"}],
                }
            },
        ),
        make_sample(
            reward=None,
            index=18,
            status=Sample.Status.ABORTED,
            metadata={
                "multi_attempt": {
                    "invalid_reason": "non_completed_attempt",
                    "attempts": [{"engine_finish_reason": "length"}],
                }
            },
        ),
    ]

    with pytest.raises(RuntimeError) as exc_info:
        collect_eval_rewards(samples, "success")

    message = str(exc_info.value)
    assert "reason=non_completed_attempt,status=aborted,finish=length:2" in message
    assert "test_prompt" not in message


def test_finalize_eval_rewards_fills_only_deferred_standard_results():
    standard_sample = make_sample(reward={"success": 1.0})
    data = {
        "standard": {"rewards": None, "samples": [standard_sample]},
        "custom": {"rewards": [0.25], "samples": []},
    }

    assert finalize_eval_rewards(data, "success") is data
    assert data["standard"]["rewards"] == [1.0]
    assert data["custom"]["rewards"] == [0.25]


def test_finalize_eval_rewards_preserves_raw_samples_when_validation_fails():
    invalid_sample = make_sample(reward=None, index=23)
    data = {"standard": {"rewards": None, "samples": [invalid_sample]}}

    with pytest.raises(RuntimeError, match=r"indices=\[23\]"):
        finalize_eval_rewards(data, "success")

    assert data["standard"]["samples"] == [invalid_sample]
    assert data["standard"]["rewards"] is None


def test_persist_then_finalize_saves_raw_samples_before_validation_failure():
    invalid_sample = make_sample(reward=None, index=29)
    data = {"standard": {"rewards": None, "samples": [invalid_sample]}}
    events = []

    def persist(raw_data):
        assert raw_data["standard"]["rewards"] is None
        assert raw_data["standard"]["samples"] == [invalid_sample]
        events.append("persisted")

    with pytest.raises(RuntimeError, match=r"indices=\[29\]"):
        persist_then_finalize_eval_rewards(data, "success", persist)

    assert events == ["persisted"]
