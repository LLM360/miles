import logging
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.training_utils.test_true_on_policy_loss_metrics import (
    _make_args,
    _make_batch,
    _patch_single_rank_loss_helpers,
)

from miles.backends.training_utils.loss_hub import diagnostics, losses
from miles.backends.training_utils.loss_hub.corrections import vanilla_tis_function


def test_nonfinite_stats_count_and_exclude_bad_values():
    assert diagnostics._nan_dbg_finite_stats(torch.tensor([float("nan"), float("inf"), -3.0, 2.0])) == (
        -3.0,
        2.0,
        3.0,
        2,
    )
    assert diagnostics._nan_dbg_finite_stats(torch.tensor([]))[3] == 0


@pytest.mark.parametrize("length,warns", [(65536, False), (65537, True)])
def test_long_batch_boundary(caplog, length, warns):
    args = SimpleNamespace(calculate_per_token_loss=False, use_dynamic_global_batch_size=False)
    with caplog.at_level(logging.WARNING):
        diagnostics._nan_dbg_warn_long_batch(args, {"response_lengths": [length], "loss_masks": [torch.ones(1)]})
    assert ("NANDBG_LONG_BATCH" in caplog.text) is warns


def test_tis_overflow_diagnostic_keeps_clipped_loss(caplog):
    pg = torch.tensor([2.0], requires_grad=True)
    args = SimpleNamespace(tis_clip_low=0.0, tis_clip=1.5)
    with caplog.at_level(logging.ERROR):
        loss, _, metrics = vanilla_tis_function(
            args,
            pg_loss=pg,
            train_log_probs=[torch.tensor([1000.0])],
            rollout_log_probs=[torch.tensor([0.0])],
            loss_masks=[],
        )
    assert loss.item() == 3.0
    loss.sum().backward()
    assert pg.grad.item() == 1.5
    assert torch.isinf(metrics["tis"]).all()
    assert "NANDBG_BAD_TIS" in caplog.text


@pytest.mark.parametrize("bad", [False, True])
def test_policy_diagnostics_observe_raw_inputs_and_preserve_gradients(monkeypatch, caplog, bad):
    args = _make_args(use_rollout_logprobs=False)
    batch = _make_batch(
        old_log_probs=torch.tensor([float("nan") if bad else -0.2, -0.3]), rollout_log_probs=torch.tensor([-0.2, -0.3])
    )
    monkeypatch.setattr(losses, "get_parallel_state", lambda: SimpleNamespace(tp=SimpleNamespace(group=None)))
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(losses, "get_log_probs_and_entropy", lambda logits, **kwargs: {"log_probs": [logits]})
    logits = torch.tensor([-0.4, -0.5], requires_grad=True)
    with caplog.at_level(logging.ERROR):
        value, metrics = losses.policy_loss_function(args, batch, logits, lambda x: x.mean())
    value.backward()
    actual_grad = logits.grad.clone()
    assert metrics["nan_dbg/nonfinite_count"].item() == (3 if bad else 0)
    assert ("name=ppo_kl" in caplog.text) is bad
    assert all(not value.requires_grad for key, value in metrics.items() if key.startswith("nan_dbg/"))
    monkeypatch.setattr(losses, "policy_nan_metrics", lambda *args: {})
    logits2 = logits.detach().clone().requires_grad_()
    baseline, _ = losses.policy_loss_function(args, batch, logits2, lambda x: x.mean())
    baseline.backward()
    assert torch.equal(value, baseline)
    assert torch.equal(actual_grad, logits2.grad)


@pytest.mark.parametrize("estimator,use_tis", [("grpo", False), ("gspo", False), ("grpo", True)])
def test_real_cpu_logprobs_and_gradients_unchanged(monkeypatch, estimator, use_tis):
    from tests.fast.backends.training_utils.loss.loss_test_utils import (
        make_args,
        make_batch,
        make_inputs,
        make_parallel_state,
    )
    from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean

    make_parallel_state()
    args = make_args(
        advantage_estimator=estimator,
        use_tis=use_tis,
        entropy_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=True,
    )
    inputs = make_inputs(seed=37, batch_size=2, prompt_lens=[3, 4], response_lens=[2, 3], vocab_size=16, args=args)
    batch = make_batch(inputs, "policy_loss")
    reducer = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"])
    logits = inputs["policy_logits"].detach().clone().requires_grad_()
    actual, metrics = losses.policy_loss_function(args, batch, logits, reducer)
    actual.backward()
    monkeypatch.setattr(losses, "policy_nan_metrics", lambda *args: {})
    baseline_logits = logits.detach().clone().requires_grad_()
    baseline, _ = losses.policy_loss_function(args, batch, baseline_logits, reducer)
    baseline.backward()
    assert torch.equal(actual, baseline)
    assert torch.equal(logits.grad, baseline_logits.grad)
    assert metrics["nan_dbg/nonfinite_count"].item() == 0
