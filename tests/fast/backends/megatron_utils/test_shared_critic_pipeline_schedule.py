"""Keep shared-critic forward schedules identical across pipeline stages."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils import actor as actor_module


class ReachedAdvantages(Exception):
    """Stop the actor after value collection, before real training starts."""


def _run_until_advantages(monkeypatch, *, last_stage, args, omit_values=False):
    calls = []
    rollout_data = {}
    actor = SimpleNamespace(
        args=args,
        model=[],
        weights_backuper=SimpleNamespace(backup_tags=[]),
        _active_model_tag="actor",
    )

    def switch_model(tag):
        actor._active_model_tag = tag

    def log_probs(*unused, collect_values=False, **kwargs):
        calls.append(("log_probs", collect_values, actor._active_model_tag))
        result = {"log_probs": [object()]} if last_stage else {}
        if last_stage and collect_values and not omit_values:
            result["values"] = [object()]
        return result

    def values(*unused):
        calls.append(("values", True, actor._active_model_tag))
        return {"values": [object()]} if last_stage and not omit_values else {}

    def reached_advantages(*unused):
        raise ReachedAdvantages

    actor._switch_model = switch_model
    actor.compute_log_prob = log_probs
    actor.compute_values = values
    monkeypatch.setattr(actor_module, "validate_rollout_for_grpo_training_step", lambda *a, **kw: None)
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda *a: ([], []))
    monkeypatch.setattr(actor_module, "all_replay_managers", [])
    monkeypatch.setattr(actor_module, "inverse_timer", lambda *a: nullcontext())
    monkeypatch.setattr(actor_module, "timer", lambda *a: nullcontext())
    monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace(is_pp_last_stage=last_stage))
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", reached_advantages)

    with pytest.raises(ReachedAdvantages):
        actor_module.MegatronTrainRayActor.train_actor(actor, 0, rollout_data)
    return calls, rollout_data


@pytest.mark.parametrize(
    "share_backbone,use_rollout_logprobs,get_mismatch_metrics,keep_old_actor,expected_calls",
    [
        (True, False, False, False, [("log_probs", True, "actor")]),
        (True, True, False, False, [("values", True, "actor")]),
        (True, True, True, False, [("log_probs", True, "actor")]),
        (True, False, False, True, [("log_probs", False, "old_actor"), ("values", True, "actor")]),
        (False, False, False, False, [("log_probs", False, "actor")]),
    ],
)
def test_forward_schedule_is_independent_of_pipeline_local_values(
    monkeypatch, share_backbone, use_rollout_logprobs, get_mismatch_metrics, keep_old_actor, expected_calls
):
    args = SimpleNamespace(
        compute_advantages_and_returns=True,
        share_backbone_critic=share_backbone,
        use_separate_critic=False,
        use_rollout_logprobs=use_rollout_logprobs,
        get_mismatch_metrics=get_mismatch_metrics,
        keep_old_actor=keep_old_actor,
    )
    for last_stage in [False, True]:
        calls, data = _run_until_advantages(monkeypatch, last_stage=last_stage, args=args)
        assert calls == expected_calls
        assert ("values" in data) == (last_stage and share_backbone)


def test_missing_shared_values_on_final_stage_fail_explicitly(monkeypatch):
    args = SimpleNamespace(
        compute_advantages_and_returns=True,
        share_backbone_critic=True,
        use_separate_critic=False,
        use_rollout_logprobs=False,
        get_mismatch_metrics=False,
        keep_old_actor=False,
    )
    with pytest.raises(AssertionError, match="did not produce values on the last PP stage"):
        _run_until_advantages(monkeypatch, last_stage=True, args=args, omit_values=True)
