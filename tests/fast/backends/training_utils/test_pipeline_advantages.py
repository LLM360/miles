"""Pipeline ownership of advantage computation, including rollout log-probs."""

from argparse import Namespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.training_utils import loss as loss_module
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


@pytest.mark.parametrize("stage", ["non-final", "final", "default-single-stage"])
def test_ppo_only_computes_advantages_on_final_pipeline_stage(monkeypatch, stage):
    group = GroupInfo(rank=0, size=1, group=None)
    # Leave is_pp_last_stage unset for the default construction used by
    # single-stage backends, including FSDP.
    state_kwargs = {} if stage == "default-single-stage" else {"is_pp_last_stage": stage == "final"}
    state = ParallelState(intra_dp=group, intra_dp_cp=group, cp=group, tp=group, **state_kwargs)
    monkeypatch.setattr(loss_module, "get_parallel_state", lambda: state)
    args = Namespace(
        use_rollout_logprobs=True,
        advantage_estimator="ppo",
        kl_coef=0.0,
        gamma=1.0,
        lambd=1.0,
        normalize_advantages=False,
        qkv_format="thd",
    )
    rollout_log_probs = torch.tensor([-0.5, -0.25])
    data = dict(
        rollout_log_probs=[rollout_log_probs],
        rewards=[1.0],
        response_lengths=[2],
        total_lengths=[3],
        loss_masks=[torch.ones(2)],
    )
    expected_advantages = [torch.tensor([0.8, 0.2])]
    expected_returns = [torch.ones(2)]
    gae = Mock(return_value=(expected_advantages, expected_returns))
    monkeypatch.setattr(loss_module, "get_advantages_and_returns_batch", gae)
    if stage != "non-final":
        data["values"] = [torch.tensor([0.2, 0.8])]

    original_keys = set(data)
    loss_module.compute_advantages_and_returns(args, data)

    if stage == "non-final":
        gae.assert_not_called()
        assert set(data) == original_keys
    else:
        gae.assert_called_once()
        assert gae.call_args.args[2] is data["values"]
        torch.testing.assert_close(gae.call_args.args[3][0], torch.tensor([0.0, 1.0]))
        assert data["advantages"] is expected_advantages
        assert data["returns"] is expected_returns
    torch.testing.assert_close(rollout_log_probs, torch.tensor([-0.5, -0.25]))
