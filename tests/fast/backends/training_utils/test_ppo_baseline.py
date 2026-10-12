"""`--ppo-baseline` hybrid baselines: leave-one-out group mean, per-mode b_t,
scale-only advantage normalization, and the CP=1 trainer path."""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss import _apply_hybrid_baseline
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.ppo_baseline import compute_group_loo_baselines, hybrid_advantages, hybrid_baseline


@pytest.fixture(autouse=True)
def _cp1_parallel_state():
    group_info = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(ParallelState(intra_dp=group_info, intra_dp_cp=group_info, cp=group_info, tp=group_info))
    yield


def test_group_loo_excludes_self_and_removed():
    ids = [0, 0, 0, 0, 1, 1, 2]
    rewards = [1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0]
    valid = [True, True, True, False, True, True, True]
    b, ok = compute_group_loo_baselines(ids, rewards, valid)
    assert b[:3] == pytest.approx([0.5, 1.0, 0.5])
    # A removed sample still gets the mean of the valid members (it is loss-masked anyway).
    assert b[3] == pytest.approx(2 / 3)
    assert b[4:6] == pytest.approx([0.0, 1.0])
    # Singleton group: no other member -> invalid.
    assert (b[6], ok[6]) == (0.0, 0)
    assert ok[:6] == [1] * 6


def test_group_loo_all_others_removed_is_invalid():
    b, ok = compute_group_loo_baselines([0, 0], [1.0, 0.0], [True, False])
    assert ok == [0, 1]
    assert b[1] == pytest.approx(1.0)


V = torch.tensor([0.2, 0.6, 0.9])


@pytest.mark.parametrize(
    "mode,cap,expected",
    [
        ("value", None, [0.2, 0.6, 0.9]),
        ("group", None, [0.5, 0.5, 0.5]),
        ("max", None, [0.5, 0.6, 0.9]),
        ("max", 0.2, [0.5, 0.6, 0.7]),
        ("min", None, [0.2, 0.5, 0.5]),
        ("min", 0.1, [0.4, 0.5, 0.5]),
        ("delta", None, [0.5, 0.9, 1.2]),
        ("delta", 0.3, [0.5, 0.8, 0.8]),
        ("v0", None, [0.2, 0.2, 0.2]),
    ],
)
def test_hybrid_baseline_modes(mode, cap, expected):
    assert hybrid_baseline(mode, V, 0.5, cap).tolist() == pytest.approx(expected)


def test_hybrid_baseline_rejects_unknown_mode():
    with pytest.raises(ValueError):
        hybrid_baseline("mean", V, 0.5)


def test_hybrid_advantages_and_diagnostics():
    returns = torch.ones(3)
    mask = torch.tensor([1, 1, 0])
    adv, mean_b, frac = hybrid_advantages("max", V, returns, mask, 0.5, True)
    assert adv.tolist() == pytest.approx([0.5, 0.4, 0.1])
    assert mean_b == pytest.approx(0.55)  # masked mean of [0.5, 0.6]
    assert frac == pytest.approx(0.5)  # critic moved the baseline on 1 of 2 masked tokens


def test_hybrid_advantages_invalid_group_falls_back_to_value():
    adv, _, frac = hybrid_advantages("group", V, torch.ones(3), torch.ones(3), 0.0, False)
    assert adv.tolist() == pytest.approx([0.8, 0.4, 0.1])
    assert frac == pytest.approx(1.0)


def test_hybrid_advantages_v0_ignores_group():
    adv, mean_b, frac = hybrid_advantages("v0", V, torch.ones(3), torch.ones(3), 0.5, False)
    assert adv.tolist() == pytest.approx([0.8, 0.8, 0.8])
    assert (mean_b, frac) == pytest.approx((0.2, 1.0))


def test_apply_hybrid_baseline_cp1():
    args = Namespace(ppo_baseline="max", ppo_baseline_cap=None, qkv_format="thd")
    values = [torch.tensor([0.2, 0.6]), torch.tensor([0.1, 0.1, 0.1])]
    returns = [torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])]
    rollout_data = {
        "group_baseline": [0.5, 0.3],
        "group_baseline_valid": [1, 1],
        "total_lengths": [5, 7],
        "response_lengths": [2, 3],
        "loss_masks": [torch.ones(2, dtype=torch.int), torch.ones(3, dtype=torch.int)],
    }
    adv = _apply_hybrid_baseline(args, rollout_data, values, returns)
    assert adv[0].tolist() == pytest.approx([0.5, 0.4])
    assert adv[1].tolist() == pytest.approx([-0.3, -0.3, -0.3])
    assert rollout_data["ppo_baseline"] == pytest.approx([0.55, 0.3])
    assert rollout_data["ppo_baseline_critic_frac"] == pytest.approx([0.5, 0.0])


def test_apply_hybrid_baseline_requires_group_field():
    args = Namespace(ppo_baseline="group", ppo_baseline_cap=None, qkv_format="thd")
    with pytest.raises(AssertionError):
        _apply_hybrid_baseline(args, {"total_lengths": [], "response_lengths": [], "loss_masks": []}, [], [])


def test_scale_only_whiten_keeps_mean(monkeypatch):
    import miles.utils.distributed_utils as du

    monkeypatch.setattr(du.dist, "all_reduce", lambda t, group=None: t)
    x = torch.tensor([1.0, 2.0, 3.0, 6.0])
    m = torch.ones(4)
    out = du.distributed_masked_whiten(x, m, center=False)
    std = x.std()  # Bessel-corrected, as in the helper
    assert out.tolist() == pytest.approx((x / std).tolist(), rel=1e-5)
    centred = du.distributed_masked_whiten(x, m)
    assert float(centred.mean()) == pytest.approx(0.0, abs=1e-6)
