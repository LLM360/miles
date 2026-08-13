import pytest
import torch

from miles.utils.ppo_utils import compute_opd_reward


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
