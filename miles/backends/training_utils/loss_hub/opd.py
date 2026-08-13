from argparse import Namespace

import torch

from miles.backends.training_utils.loss_hub.math_utils import compute_opd_reward
from miles.utils.types import RolloutBatch


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of old-student log-probability tensors. OPD
            treats these as fixed scoring inputs.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
        return

    precomputed_reverse_kls = rollout_data.get("opd_reverse_kl")
    reward_type = getattr(args, "opd_reward_type", "logr")
    if reward_type not in {"logr", "k3"}:
        raise ValueError(f"Unknown OPD reward type: {reward_type}")
    if reward_type == "k3" and getattr(args, "opd_log_prob_top_k", 0) > 0:
        raise ValueError("k3 OPD requires sampled teacher/student log-probs, not precomputed top-k rewards")
    if precomputed_reverse_kls is not None and reward_type == "logr":
        if len(advantages) != len(precomputed_reverse_kls):
            raise ValueError(
                f"OPD length mismatch: advantages={len(advantages)}, "
                f"opd_reverse_kl={len(precomputed_reverse_kls)}."
            )

        reverse_kls = []
        for i, adv in enumerate(advantages):
            reverse_kl = precomputed_reverse_kls[i]
            if not torch.is_tensor(reverse_kl):
                reverse_kl = torch.tensor(reverse_kl, dtype=torch.float32)
            # Defensive consumer boundary for direct callers that bypass
            # compute_advantages_and_returns' persistent-data detach.
            reverse_kl = reverse_kl.detach().to(device=adv.device)
            if adv.shape != reverse_kl.shape:
                raise ValueError(
                    f"OPD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                    f"opd_reverse_kl={tuple(reverse_kl.shape)}."
                )
            advantages[i] = adv - args.opd_kl_coef * reverse_kl
            reverse_kls.append(reverse_kl)

        rollout_data["opd_reverse_kl"] = reverse_kls
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    if not (len(advantages) == len(student_log_probs) == len(teacher_log_probs)):
        raise ValueError(
            f"OPD length mismatch: advantages={len(advantages)}, "
            f"student_log_probs={len(student_log_probs)}, teacher_log_probs={len(teacher_log_probs)}."
        )

    device = student_log_probs[0].device
    detached_teacher_log_probs = [t.detach() for t in teacher_log_probs]
    rollout_data["teacher_log_probs"] = detached_teacher_log_probs
    teacher_log_probs = [t.to(device=device) for t in detached_teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        if student_log_probs[i].shape != teacher_log_probs[i].shape:
            raise ValueError(
                f"OPD shape mismatch at sample {i}: student_log_probs={tuple(student_log_probs[i].shape)}, "
                f"teacher_log_probs={tuple(teacher_log_probs[i].shape)}."
            )
        if adv.shape != student_log_probs[i].shape:
            raise ValueError(
                f"OPD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                f"student_log_probs={tuple(student_log_probs[i].shape)}. "
                "OPD expects per-token advantages; broadcast scalar advantages must be expanded before this call."
            )
        old_student_log_prob = student_log_probs[i].detach()
        reverse_kl = (
            -compute_opd_reward(old_student_log_prob, teacher_log_probs[i], "k3")
            if reward_type == "k3"
            else old_student_log_prob - teacher_log_probs[i]
        )
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging.
    rollout_data["opd_reverse_kl"] = reverse_kls


def compute_legacy_opd_advantages(args, rollout_data, student_log_probs):
    """Stable's OPD-only estimator on response/CP-aligned fixed scoring inputs.

    It ignores scalar rewards and the additive OPD coefficient, exactly as the
    original estimator did. Current rollout conversion aligns teacher scores.
    """
    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if student_log_probs is None or teacher_log_probs is None:
        raise ValueError("on_policy_distillation requires student and teacher log-probs")
    if len(student_log_probs) != len(teacher_log_probs):
        raise ValueError("OPD student/teacher sample counts differ")
    advantages = []
    for student, teacher in zip(student_log_probs, teacher_log_probs, strict=True):
        if student.shape != teacher.shape:
            raise ValueError("Legacy OPD requires response-aligned teacher scores on each context-parallel rank")
        advantages.append(
            compute_opd_reward(
                student.detach(), teacher.detach().to(device=student.device), getattr(args, "opd_reward_type", "logr")
            )
        )
    return advantages
