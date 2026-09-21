from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.ppo_utils import compute_opd_reward, fill_monte_carlo_returns


def test_compute_opd_reward_logr() -> None:
    student = torch.tensor([-2.0, -1.0])
    teacher = torch.tensor([-1.5, -1.25])

    actual = compute_opd_reward(student, teacher, "logr")

    torch.testing.assert_close(actual, torch.tensor([0.5, -0.25]))


def test_compute_opd_reward_k3_is_negative_kl_estimate() -> None:
    student = torch.tensor([-2.0, -1.0])
    teacher = torch.tensor([-1.5, -1.25])
    log_r = teacher - student
    expected = 1 + log_r - torch.exp(log_r)

    actual = compute_opd_reward(student, teacher, "k3")

    torch.testing.assert_close(actual, expected)
    assert torch.all(actual <= 0)


def test_compute_opd_reward_k3_is_zero_when_distributions_match() -> None:
    log_probs = torch.tensor([-10.0, -1.0, 0.0])

    actual = compute_opd_reward(log_probs, log_probs, "k3")

    torch.testing.assert_close(actual, torch.zeros_like(log_probs))


def test_compute_opd_reward_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown OPD reward type"):
        compute_opd_reward(torch.tensor([0.0]), torch.tensor([0.0]), "unknown")


def test_fill_monte_carlo_returns_matches_episode_reward() -> None:
    group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(ParallelState(intra_dp=group, intra_dp_cp=group, cp=group, tp=group))
    args = Namespace(qkv_format="thd")
    data = dict(
        rewards=[0.0, 1.0],
        response_lengths=[2, 3],
        total_lengths=[5, 6],
        loss_masks=[torch.tensor([0, 1], dtype=torch.int), torch.ones(3, dtype=torch.int)],
    )
    fill_monte_carlo_returns(args, data)
    assert [t.tolist() for t in data["returns"]] == [[0.0, 0.0], [1.0, 1.0, 1.0]]
    assert [t.tolist() for t in data["advantages"]] == [[0.0, 0.0], [0.0, 0.0, 0.0]]


def test_fill_monte_carlo_returns_broadcasts_scalar_reward() -> None:
    group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(ParallelState(intra_dp=group, intra_dp_cp=group, cp=group, tp=group))
    args = Namespace(qkv_format="thd")
    data = dict(
        rewards=[0.4],
        response_lengths=[4],
        total_lengths=[7],
        loss_masks=[torch.tensor([0, 1, 1, 0], dtype=torch.int)],
    )
    fill_monte_carlo_returns(args, data)
    torch.testing.assert_close(data["returns"][0], torch.full((4,), 0.4))
    torch.testing.assert_close(data["advantages"][0], torch.zeros(4))
