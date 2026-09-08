"""Numerical and gradient contracts for domain diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tests.fast.backends.training_utils.test_true_on_policy_loss_metrics import _make_args

from miles.backends.training_utils import cp_utils, log_utils
from miles.backends.training_utils.domain_metrics import compute_domain_metrics
from miles.backends.training_utils.loss_hub import losses, math_utils
from miles.backends.training_utils.metric_schema import sum_aligned_metrics


@pytest.fixture
def state(monkeypatch):
    state = SimpleNamespace(
        cp=SimpleNamespace(size=1, rank=0),
        effective_dp_cp=SimpleNamespace(size=1, groups_inner_to_outer=[]),
    )
    for module in (cp_utils, losses, log_utils, math_utils):
        monkeypatch.setattr(module, "get_parallel_state", lambda: state)
    return state


def _args(**overrides):
    args = _make_args(use_rollout_logprobs=False)
    vars(args).update(loss_agg_mode="sample-mean", **overrides)
    return args


def _batch():
    return {
        "domains": ["math", "code"],
        "all_domains": ["code", "math", "unseen"],
        "response_lengths": [2, 1],
        "total_lengths": [2, 1],
        "loss_masks": [torch.ones(2), torch.ones(1)],
        "rollout_mask_sums": [torch.tensor(2.0), torch.tensor(1.0)],
        "unconcat_tokens": [torch.tensor([1, 2]), torch.tensor([3])],
        "advantages": [torch.tensor([1.0, -0.5]), torch.tensor([0.2])],
        "log_probs": [torch.tensor([-0.5, -0.9]), torch.tensor([-0.2])],
        "rollout_log_probs": [torch.tensor([-0.7, -0.8]), torch.tensor([-0.3])],
        "ref_log_probs": [torch.tensor([-0.6, -0.8]), torch.tensor([-0.4])],
    }


@pytest.mark.parametrize("mode", ["sample-mean", "token-mean", "token-sum"])
@pytest.mark.parametrize("entropy", [False, True])
@pytest.mark.parametrize("use_kl", [False, True])
@pytest.mark.parametrize("rejection", [False, True])
def test_domain_metrics_leave_loss_gradients_and_parent_metrics_unchanged(
    state, monkeypatch, mode, entropy, use_kl, rejection
):
    args = _args(entropy_coef=0.1 if entropy else 0.0, use_kl_loss=use_kl, kl_loss_coef=0.2, use_tis=rejection)
    args.loss_agg_mode = mode
    batch = _batch()

    def scores(logits, *, with_entropy, **kwargs):
        chunks = list(logits.split([2, 1]))
        return {"log_probs": chunks, **({"entropy": [x.square() for x in chunks]} if with_entropy else {})}

    def reject(*, pg_loss, loss_masks, **kwargs):
        masks = [mask.clone() for mask in loss_masks]
        masks[0][0] = 0
        return pg_loss * 0.8, masks, {"tis_clipfrac": torch.ones_like(pg_loss)}

    monkeypatch.setattr(losses, "get_log_probs_and_entropy", scores)
    if rejection:
        args.custom_tis_function_path = "test.reject"
        monkeypatch.setattr(losses, "load_function", lambda path: reject)
    reducer = cp_utils.get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        denominators=batch["rollout_mask_sums"],
        loss_agg_mode=mode,
    )
    x = torch.tensor([-0.4, -1.1, -0.25], requires_grad=True)
    original_batch = {key: val for key, val in batch.items() if key not in ("domains", "all_domains")}
    original_loss, original_metrics = losses.policy_loss_function(args, original_batch, x, reducer)
    original_grad = torch.autograd.grad(original_loss, x)[0]
    y = x.detach().clone().requires_grad_()
    domain_loss, metrics = losses.policy_loss_function(args, batch, y, reducer)
    torch.testing.assert_close(domain_loss, original_loss, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(domain_loss, y)[0], original_grad, rtol=0, atol=0)
    for name, value in original_metrics.items():
        # Batch-wide NaN ranges/counts are diagnostics, not additive domain contributions.
        is_nan_diagnostic = name.startswith("nan_dbg/")
        torch.testing.assert_close(metrics[name], value, rtol=0, atol=0, equal_nan=is_nan_diagnostic)
        if name not in ("ess_ratio",) and not is_nan_diagnostic:
            torch.testing.assert_close(metrics[f"{name}/math"] + metrics[f"{name}/code"], value)
            assert metrics[f"{name}/unseen"].item() == 0
    assert all(not value.requires_grad for name, value in metrics.items() if "/" in name)
    if rejection:
        # The rejected token still contributes to the pre-rejection mismatch diagnostic.
        assert metrics["tis_clipfrac/math"].item() == (1 if mode == "sample-mean" else 2)


def test_domain_contributions_keep_global_normalization(state):
    args = _args()
    batch = _batch()
    values = torch.tensor([2.0, 2.0, 8.0])
    metrics = compute_domain_metrics(
        args, batch, {"pg_loss": values, "entropy_loss": torch.zeros(3)}, loss_masks=batch["loss_masks"]
    )
    row = {"keys": list(metrics), "values": torch.tensor([2, *metrics.values()])}
    result = log_utils.aggregate_train_losses([row])
    assert result["loss/math"] == 1
    assert result["loss/code"] == 4
    assert result["loss/unseen"] == 0


def test_sibling_samples_keep_rollout_denominator_and_inactive_nan_is_zero(state):
    args, batch = _args(), _batch()
    batch.update(domains=["math", "math"], rollout_mask_sums=[torch.tensor(3.0)] * 2)
    per_token = {"pg_loss": torch.tensor([2.0, 8.0, 8.0]), "entropy_loss": torch.zeros(3)}
    result = compute_domain_metrics(args, batch, per_token, loss_masks=batch["loss_masks"])
    assert result["loss/math"].item() == 6
    per_token["pg_loss"][:] = float("nan")
    batch["domains"] = [None, None]
    result = compute_domain_metrics(args, batch, per_token, loss_masks=batch["loss_masks"])
    assert all(value.item() == 0 for value in result.values())


def test_custom_reducer_diagnostics_do_not_claim_to_decompose_custom_loss(state):
    args, batch = _args(custom_pg_loss_reducer_function_path="custom.non_additive"), _batch()
    result = compute_domain_metrics(
        args, batch, {"pg_loss": torch.ones(3), "entropy_loss": torch.zeros(3)}, loss_masks=batch["loss_masks"]
    )
    assert "standard_pg_loss/math" in result
    assert "pg_loss/math" not in result
    assert "loss/math" not in result


@pytest.mark.parametrize("mode", ["sample-mean", "token-mean", "token-sum"])
def test_context_parallel_chunks_reconstruct_domain_contribution(state, mode):
    args = _args()
    args.loss_agg_mode = mode
    state.cp.size = 2
    batch = {
        "domains": ["math"],
        "all_domains": ["math", "code"],
        "total_lengths": [8],
        "response_lengths": [4],
        "loss_masks": [torch.tensor([1.0, 0.0, 1.0, 1.0])],
        "rollout_mask_sums": [torch.tensor(3.0)],
    }
    contributions = []
    for rank, values in [(0, [8.0]), (1, [2.0, float("nan"), 6.0])]:
        state.cp.rank = rank
        x = torch.tensor(values)
        metrics = compute_domain_metrics(
            args, batch, {"pg_loss": x, "entropy_loss": torch.zeros_like(x)}, loss_masks=batch["loss_masks"]
        )
        contributions.append(metrics["loss/math"])
        assert metrics["loss/code"].item() == 0
    assert sum(contributions).item() == pytest.approx(16 / 3 if mode == "sample-mean" else 16)


def test_microbatch_key_order_and_optional_keys_are_aligned(state):
    rows = [
        {"keys": ["loss/math", "loss"], "values": torch.tensor([1.0, 2.0, 2.0])},
        {"keys": ["loss", "loss/code", "ref_kl/code"], "values": torch.tensor([1.0, 8.0, 8.0, 0.4])},
    ]
    result = log_utils.aggregate_train_losses(rows)
    assert result == pytest.approx({"loss": 5, "loss/math": 1, "loss/code": 4, "ref_kl/code": 0.2})


def test_duplicate_metric_names_fail_instead_of_silent_aliasing():
    with pytest.raises(ValueError, match="unique"):
        sum_aligned_metrics([{"keys": ["loss", "loss"], "values": torch.ones(3)}], ["loss"])


def test_array_backed_domain_labels_are_supported(state):
    args, batch = _args(), _batch()
    batch["domains"] = np.array(batch["domains"], dtype=object)
    batch["all_domains"] = np.array(batch["all_domains"], dtype=object)
    result = compute_domain_metrics(
        args, batch, {"pg_loss": torch.ones(3), "entropy_loss": torch.zeros(3)}, loss_masks=batch["loss_masks"]
    )
    assert result["loss/math"].item() == 1
    assert result["loss/code"].item() == 1
