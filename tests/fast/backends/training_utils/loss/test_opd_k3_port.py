from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub.math_utils import compute_opd_reward
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages


@pytest.mark.parametrize("reward_type", ["logr", "k3"])
def test_legacy_estimator_uses_only_token_reward_once(reward_type):
    student = torch.tensor([-2.0, -1.0], requires_grad=True)
    teacher = torch.tensor([-1.5, -1.25], requires_grad=True)
    args = Namespace(
        skip_actor_forward_only=False,
        use_rollout_logprobs=False,
        kl_coef=0,
        advantage_estimator="on_policy_distillation",
        use_opd=True,
        opd_type="sglang",
        opd_reward_type=reward_type,
        opd_kl_coef=999,
        normalize_advantages=False,
    )
    data = dict(
        log_probs=[student],
        teacher_log_probs=[teacher],
        rewards=[99],
        response_lengths=[2],
        total_lengths=[3],
        loss_masks=[torch.ones(2)],
    )
    compute_advantages_and_returns(args, data)
    expected = compute_opd_reward(student.detach(), teacher.detach(), reward_type)
    torch.testing.assert_close(data["advantages"][0], expected)
    torch.testing.assert_close(data["returns"][0], expected)
    assert not data["advantages"][0].requires_grad


def test_additive_k3_keeps_base_advantages_and_detaches_inputs():
    student = [torch.tensor([-2.0, -1.0], requires_grad=True)]
    teacher = [torch.tensor([-1.5, -1.25], requires_grad=True)]
    args = Namespace(opd_type="sglang", opd_reward_type="k3", opd_kl_coef=0.5)
    data = {"teacher_log_probs": teacher}
    reward = compute_opd_reward(student[0].detach(), teacher[0].detach(), "k3")
    # Repeat on the same persistent rollout data: stored metrics are not a top-k input.
    for _ in range(2):
        advantages = [torch.tensor([3.0, 4.0])]
        apply_opd_kl_to_advantages(args, data, advantages, student)
        torch.testing.assert_close(advantages[0], torch.tensor([3.0, 4.0]) + 0.5 * reward)
        torch.testing.assert_close(data["opd_reverse_kl"][0], -reward)
        assert not advantages[0].requires_grad


def test_k3_cannot_use_topk_averaged_ratios():
    args = Namespace(opd_type="sglang", opd_reward_type="k3", opd_kl_coef=1, opd_log_prob_top_k=4)
    with pytest.raises(ValueError, match="sampled"):
        apply_opd_kl_to_advantages(args, {"opd_reverse_kl": [torch.ones(2)]}, [torch.zeros(2)], [torch.zeros(2)])


@pytest.mark.parametrize("response_length", [0, 2])
def test_legacy_opd_teacher_scores_trim_before_cp_distribution(response_length):
    from tests.fast.ray.rollout.conftest import make_args, make_sample

    from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data

    sample = make_sample(response_length=response_length)
    values = [-float(i) for i in range(len(sample.tokens))]
    sample.teacher_log_probs = values
    output = convert_samples_to_train_data(
        make_args(advantage_estimator="on_policy_distillation", rewards_normalization=False),
        [sample],
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    assert output["teacher_log_probs"] == [values[-response_length:] if response_length else []]
    assert sample.teacher_log_probs is values
