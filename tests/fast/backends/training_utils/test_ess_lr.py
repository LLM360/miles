"""Fast unit tests for ESS-guided LR scaling (VCPO-style; arXiv:2602.17616).

Covers the pure ESS->lr_scale math (``ess_lr_scale_from_sums``) and the
default-off no-op path of ``ess_lr_compute``. The distributed all-reduce
plumbing inside ``ess_lr_compute`` is exercised by integration runs, not here.
"""

import math
import types
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

from miles.backends.training_utils.ess_lr import (  # noqa: E402
    _ESS_LR_STATE,
    ess_lr_compute,
    ess_lr_scale_from_sums,
)


def _sums(weights):
    w = torch.tensor(weights, dtype=torch.float32)
    return w.sum(), (w * w).sum(), torch.tensor(float(w.numel()))


def test_equal_weights_give_scale_one():
    # ESS == B  ->  rho == 1  ->  lr_scale == 1 (no shrink, fully on-policy).
    sum_w, sum_w2, b = _sums([1.0, 1.0, 1.0, 1.0])
    scale, rho = ess_lr_scale_from_sums(sum_w, sum_w2, b, floor=0.1)
    assert rho == pytest.approx(1.0, abs=1e-6)
    assert scale == pytest.approx(1.0, abs=1e-6)


def test_skewed_weights_shrink_lr():
    # One heavier weight collapses ESS below B  ->  rho < 1  ->  scale = sqrt(rho).
    weights = [1.0, 1.0, 1.0, 5.0]
    sum_w, sum_w2, b = _sums(weights)
    scale, rho = ess_lr_scale_from_sums(sum_w, sum_w2, b, floor=0.1)
    expected_rho = (8.0**2) / (4.0 * 28.0)  # 64 / 112
    assert rho == pytest.approx(expected_rho, rel=1e-5)
    assert scale == pytest.approx(math.sqrt(expected_rho), rel=1e-5)
    assert 0.1 < scale < 1.0


def test_floor_clamps_extreme_collapse():
    # rho = 1/B = 0.005  ->  sqrt(rho) ~= 0.0707 < floor  ->  clamped to floor.
    sum_w = torch.tensor(1.0)
    sum_w2 = torch.tensor(1.0)
    b = torch.tensor(200.0)
    scale, rho = ess_lr_scale_from_sums(sum_w, sum_w2, b, floor=0.1)
    assert rho == pytest.approx(0.005, rel=1e-3)
    assert math.sqrt(rho) < 0.1
    assert scale == pytest.approx(0.1, abs=1e-6)


def test_scale_never_exceeds_one():
    # rho can numerically exceed 1 with tiny denominators; scale must stay <= 1.
    sum_w = torch.tensor(10.0)
    sum_w2 = torch.tensor(1.0)
    b = torch.tensor(1.0)
    scale, _ = ess_lr_scale_from_sums(sum_w, sum_w2, b, floor=0.1)
    assert scale <= 1.0


def test_compute_is_noop_when_disabled():
    # Default off must not touch _ESS_LR_STATE (bit-exact legacy path).
    _ESS_LR_STATE["scale"] = 1.0
    _ESS_LR_STATE["rho_ess"] = 1.0
    args = Namespace(use_ess_lr=False)
    rollout_data = {
        "log_probs": [torch.zeros(4)],
        "rollout_log_probs": [torch.zeros(4)],
        "loss_masks": [torch.ones(4)],
    }
    ess_lr_compute(args, parallel_state=None, rollout_data=rollout_data)
    assert _ESS_LR_STATE["scale"] == 1.0
    assert _ESS_LR_STATE["rho_ess"] == 1.0


def _fake_parallel_state(cp_size=1, dp_size=1):
    # cp_size == 1 skips the CP all-reduce; dp_size == 1 skips the DP all-reduce,
    # so the math runs locally on one rank with no process groups needed.
    return types.SimpleNamespace(
        cp=types.SimpleNamespace(size=cp_size, group=None),
        intra_dp=types.SimpleNamespace(size=dp_size, group=None),
    )


def test_compute_updates_state_when_enabled():
    # Single rank (cp=dp=1) -> no all-reduce. Two trajectories with IS weights
    # w = [1, 5]  ->  rho = 36 / (2 * 26) = 0.6923, scale = sqrt(rho).
    _ESS_LR_STATE["scale"] = 0.5  # poison: must be overwritten
    _ESS_LR_STATE["rho_ess"] = 0.5
    args = Namespace(use_ess_lr=True, ess_lr_floor=0.1, qkv_format="thd")
    rollout_data = {
        "log_probs": [torch.zeros(4), torch.full((4,), math.log(5.0))],
        "rollout_log_probs": [torch.zeros(4), torch.zeros(4)],
        "loss_masks": [torch.ones(4), torch.ones(4)],
        "total_lengths": [4, 4],
        "response_lengths": [4, 4],
    }
    ess_lr_compute(args, _fake_parallel_state(cp_size=1), rollout_data)
    expected_rho = 36.0 / (2.0 * 26.0)
    assert _ESS_LR_STATE["rho_ess"] == pytest.approx(expected_rho, rel=1e-4)
    assert _ESS_LR_STATE["scale"] == pytest.approx(math.sqrt(expected_rho), rel=1e-4)


def test_compute_resets_when_inputs_missing():
    # Enabled but rollout_log_probs absent (e.g. the --use-rollout-logprobs path,
    # the critic path, or a non-last PP stage): must reset to a no-op scale this
    # rollout, not silently reuse the previous rollout's value.
    _ESS_LR_STATE["scale"] = 0.42
    _ESS_LR_STATE["rho_ess"] = 0.17
    args = Namespace(use_ess_lr=True, ess_lr_floor=0.1, qkv_format="thd")
    rollout_data = {"log_probs": [torch.zeros(4)], "loss_masks": [torch.ones(4)]}  # no rollout_log_probs
    ess_lr_compute(args, _fake_parallel_state(cp_size=1), rollout_data)
    assert _ESS_LR_STATE["scale"] == 1.0
    assert _ESS_LR_STATE["rho_ess"] == 1.0
