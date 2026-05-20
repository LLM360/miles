import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.backends.training_utils import cp_utils, rollout_validation
from miles.backends.training_utils.rollout_validation import validate_rollout_for_grpo_training_step as validate


def _args(**kwargs):
    return SimpleNamespace(**dict(dict(qkv_format="thd", n_samples_per_prompt=2), **kwargs))


def _data():
    return dict(
        tokens=[torch.arange(8), torch.arange(4)],
        rewards=[0.25, -0.25],
        total_lengths=[8, 4],
        response_lengths=[3, 1],
        loss_masks=[[1, 0, 1], [1]],
        rollout_log_probs=[[-0.1] * 3, [-0.3]],
    )


@pytest.fixture(autouse=True)
def parallel_state(monkeypatch):
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), is_ulysses_cp=False)
    monkeypatch.setattr(rollout_validation, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    return state


@pytest.mark.parametrize("container", [list, tuple, np.asarray, torch.as_tensor])
def test_sequence_containers_are_accepted_without_mutation(container, caplog):
    data = _data()
    for key in ["rewards", "total_lengths", "response_lengths"]:
        data[key] = container(data[key])
    data["loss_masks"] = [container(value) for value in data["loss_masks"]]
    original = copy.deepcopy(data)
    validate(_args(), data, rollout_id=4)
    assert "success" in caplog.text and "rollout_id=4" in caplog.text
    for key in data:
        for value, before in zip(data[key], original[key], strict=True):
            np.testing.assert_array_equal(value, before)


def test_zero_masks_weighted_masks_and_empty_local_responses_warn(caplog):
    data = _data()
    data["loss_masks"] = [[0, 0, 0], [2.5]]
    validate(_args(), data)
    assert "no active tokens" in caplog.text and "weighted mask" in caplog.text
    data["response_lengths"][0] = 0
    data["loss_masks"][0] = []
    data["rollout_log_probs"][0] = []
    validate(_args(), data)
    validate(_args(), {key: [] for key in data})


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("rewards", [np.nan, 0], "not finite"),
        ("rewards", "bad", "must be a sequence"),
        ("response_lengths", [9, 1], "invalid lengths"),
        ("response_lengths", [1.5, 1], "not an integer"),
        ("total_lengths", [0, 4], "invalid lengths"),
        ("loss_masks", [[1], [1]], "length 1 != expected 3"),
        ("loss_masks", [[[1, 1, 1]], [1]], "must be 1D"),
        ("loss_masks", [[1, float("inf"), 1], [1]], "NaN/Inf"),
        ("tokens", [[1, 2], [1, 2, 3, 4]], "length 2 != expected 8"),
        ("max_seq_lens", [7, 4], "< total_lengths"),
        ("rollout_log_probs", [[-0.1] * 3], "length mismatch"),
        ("teacher_log_probs", [[-0.1, -0.1, float("nan")], [-0.1]], "NaN/Inf"),
    ],
)
def test_invalid_data_reports_field_before_raising(field, value, message, caplog):
    data = _data()
    data[field] = value
    with pytest.raises(ValueError, match="rollout validation failed"):
        validate(_args(), data)
    assert message in caplog.text and "summary_before_failure" in caplog.text


@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("cp_rank", range(4))
def test_cp_checks_actual_local_length_including_empty_slices(
    parallel_state, qkv_format, cp_rank, allgather_cp, caplog
):
    parallel_state.cp.size = 4
    parallel_state.cp.rank = cp_rank
    data = _data()
    if qkv_format == "bshd":
        data["max_seq_lens"] = [16, 16]
    for key in ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl", "log_probs", "ref_log_probs", "values"):
        data[key] = [
            cp_utils.slice_log_prob_with_cp(
                [-0.2] * response, total, response, qkv_format, 16 if qkv_format == "bshd" else None
            )
            for total, response in zip(data["total_lengths"], data["response_lengths"], strict=True)
        ]
    validate(_args(qkv_format=qkv_format, allgather_cp=allgather_cp), data)
    data["rollout_log_probs"][0].append(-0.1)
    with pytest.raises(ValueError):
        validate(_args(qkv_format=qkv_format, allgather_cp=allgather_cp), data)
    assert "rollout_log_probs[0] length" in caplog.text


def test_missing_required_and_requested_logprob_fields(caplog):
    with pytest.raises(ValueError):
        validate(_args(), {})
    assert "missing required key" in caplog.text
    with pytest.raises(ValueError):
        validate(_args(), _data(), require_log_probs=True)
    validate(_args(use_rollout_logprobs=True), _data(), require_log_probs=True)
