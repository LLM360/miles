import pytest
import torch

from miles.utils.ppo_utils import compute_group_advantages, compute_importance_weighted_entropy, compute_opd_reward


def test_compute_group_advantages_preserves_grpo_mean_baseline() -> None:
    rewards = torch.tensor([[0.0, 1.0, 3.0], [2.0, 4.0, 6.0]])

    advantages, baseline = compute_group_advantages(rewards, normalize_by_std=False)

    torch.testing.assert_close(baseline, rewards.mean(dim=-1, keepdim=True))
    torch.testing.assert_close(advantages, rewards - rewards.mean(dim=-1, keepdim=True))


def test_compute_group_advantages_qae_uses_right_continuous_order_statistic() -> None:
    # At K=0.5 with four binary samples, the empirical inverse CDF selects the
    # second order statistic (zero), not torch.quantile's interpolated 0.5.
    rewards = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    advantages, baseline = compute_group_advantages(rewards, quantile=0.5, normalize_by_std=False)

    torch.testing.assert_close(baseline, torch.tensor([[0.0]]))
    torch.testing.assert_close(advantages, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))


def test_compute_group_advantages_qae_selects_only_failures_for_easy_group() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

    advantages, baseline = compute_group_advantages(rewards, quantile=0.4, normalize_by_std=False)

    torch.testing.assert_close(baseline, torch.tensor([[1.0]]))
    torch.testing.assert_close(advantages, rewards - 1.0)


def test_compute_group_advantages_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="QAE quantile"):
        compute_group_advantages(torch.tensor([[0.0, 1.0]]), quantile=1.0)


def test_compute_importance_weighted_entropy() -> None:
    log_probs = torch.log(torch.tensor([0.25, 0.5]))
    old_log_probs = torch.log(torch.tensor([0.5, 0.25]))
    expected = -log_probs * torch.tensor([0.5, 2.0])

    actual = compute_importance_weighted_entropy(log_probs, old_log_probs)

    torch.testing.assert_close(actual, expected)


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
