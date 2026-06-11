from unittest.mock import MagicMock

import pytest

from miles.rollout.generate_utils.sample_utils import (
    _merge_sample_pair,
    drop_samples_after_first_non_completed,
    merge_samples,
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


class TestDropSamplesAfterFirstNonCompleted:
    def _statuses(self, *statuses):
        return [make_sample(status=status, index=i) for i, status in enumerate(statuses)]

    def test_all_completed_unchanged(self):
        samples = self._statuses(
            Sample.Status.COMPLETED, Sample.Status.COMPLETED, Sample.Status.COMPLETED
        )
        kept, dropped = drop_samples_after_first_non_completed(samples)
        assert kept == samples
        assert dropped == 0

    def test_non_completed_final_turn_unchanged(self):
        samples = self._statuses(Sample.Status.COMPLETED, Sample.Status.ABORTED)
        kept, dropped = drop_samples_after_first_non_completed(samples)
        assert kept == samples
        assert dropped == 0

    def test_mid_session_aborted_drops_trailing_turns(self):
        samples = self._statuses(
            Sample.Status.COMPLETED,
            Sample.Status.ABORTED,
            Sample.Status.COMPLETED,
            Sample.Status.COMPLETED,
        )
        kept, dropped = drop_samples_after_first_non_completed(samples)
        assert kept == samples[:2]
        assert dropped == 2

    def test_mid_session_truncated_drops_trailing_turns(self):
        samples = self._statuses(
            Sample.Status.COMPLETED, Sample.Status.TRUNCATED, Sample.Status.COMPLETED
        )
        kept, dropped = drop_samples_after_first_non_completed(samples)
        assert kept == samples[:2]
        assert dropped == 1

    def test_first_turn_aborted_drops_all_trailing_turns(self):
        samples = self._statuses(Sample.Status.ABORTED, Sample.Status.COMPLETED)
        kept, dropped = drop_samples_after_first_non_completed(samples)
        assert kept == samples[:1]
        assert dropped == 1

    def test_single_sample_unchanged(self):
        samples = self._statuses(Sample.Status.ABORTED)
        kept, dropped = drop_samples_after_first_non_completed(samples)
        assert kept == samples
        assert dropped == 0


class TestMergeSamplesAfterAbortedTurn:
    """Regression: the engine aborted turn 2 mid-decode (e.g. end-of-rollout
    abort_request), but the agent kept going and produced a COMPLETED turn 3.
    merge_samples cannot represent a non-final aborted turn."""

    def _make_session_turns(self):
        turn1 = make_sample(
            tokens=[1, 2, 10, 11],
            response="turn1",
            response_length=2,
            loss_mask=[1, 1],
        )
        turn2 = make_sample(
            tokens=[1, 2, 10, 11, 50, 20, 21],
            response="turn2-partial",
            response_length=2,
            loss_mask=[1, 1],
            status=Sample.Status.ABORTED,
        )
        turn3 = make_sample(
            tokens=[1, 2, 10, 11, 50, 20, 21, 60, 30, 31],
            response="turn3",
            response_length=2,
            loss_mask=[1, 1],
        )
        return [turn1, turn2, turn3]

    def test_merge_without_drop_raises(self, mock_tokenizer):
        with pytest.raises(AssertionError, match="a.status must be COMPLETED"):
            merge_samples(self._make_session_turns(), mock_tokenizer)

    def test_drop_then_merge_ends_aborted(self, mock_tokenizer):
        kept, dropped = drop_samples_after_first_non_completed(self._make_session_turns())

        assert dropped == 1
        merged = merge_samples(kept, mock_tokenizer)
        assert merged.status == Sample.Status.ABORTED
        assert merged.response_length == 2 + 1 + 2  # turn1 + obs + turn2
        assert "turn2-partial" in merged.response
