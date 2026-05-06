"""Tests for per-domain metric fan-out and unified correctness signal.

These exercise the math/logic added when upstreaming OPD-specific metrics
into miles. The goal is to verify:

1. The masked-loss-mask trick used for per-domain reductions: when we zero
   out non-target samples' loss_masks, the resulting `sum_of_sample_mean`
   reducer produces the same value as `sum_of_sample_mean` over only the
   target-domain samples.

2. Per-domain reductions partition the global reduction. When domains
   partition the batch and every sample has the same per-token weight
   distribution, the (count-weighted) sum across domains matches the
   global reduction.

3. The `_correctness(s, args)` helper unifies scalar GRPO reward sign and
   non-scalar OPD `metadata["correctness_reward"]` into one path.

4. `compute_samples_seen` returns the cumulative per-rollout sample count.
"""
from __future__ import annotations

import sys
import types
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.utils.metric_utils import compute_samples_seen


# ---------------------------------------------------------------------------
# Stub miles.backends.training_utils.parallel.get_parallel_state so cp_utils'
# sum_of_sample_mean (which calls it) returns CP=1.
# ---------------------------------------------------------------------------

class _FakePG:
    size = 1
    rank = 0


class _FakeParallelState:
    cp = _FakePG()


def _patch_parallel_state(monkeypatch):
    import miles.backends.training_utils.cp_utils as cp_utils
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: _FakeParallelState())


# ---------------------------------------------------------------------------
# 1. Masked-loss-mask trick: per-domain reducer ignores non-target samples
# ---------------------------------------------------------------------------

def test_domain_filtered_reducer_matches_per_domain_subset(monkeypatch):
    _patch_parallel_state(monkeypatch)

    # 3 samples: math, code, math. Each sample has 4 response tokens.
    # Per-token "values" tensor `x` is the per-sample value broadcast over tokens.
    domains = ["math", "code", "math"]
    response_lengths = [4, 4, 4]
    total_lengths = [4, 4, 4]
    loss_masks = [torch.ones(4) for _ in range(3)]

    # x: per-sample mean is [1.0, 2.0, 3.0]
    x = torch.tensor([1.0]*4 + [2.0]*4 + [3.0]*4)

    # Build a "math"-filtered reducer
    masked_for_math = [
        lm if d == "math" else torch.zeros_like(lm)
        for d, lm in zip(domains, loss_masks)
    ]
    math_reducer = get_sum_of_sample_mean(
        total_lengths, response_lengths, masked_for_math,
        calculate_per_token_loss=False, qkv_format="thd", max_seq_lens=None,
    )

    # sum_of_sample_mean: per-sample token-mean, then sum across samples.
    # For math: sample 0 contributes 1.0, sample 2 contributes 3.0, sample 1
    # contributes 0 (its mask is all zeros, clamp_min returns 1 in denominator
    # but numerator is also 0). Result should be 1.0 + 0 + 3.0 = 4.0.
    result = math_reducer(x).item()
    assert result == 4.0, f"expected 4.0, got {result}"

    # Same for "code": only sample 1 contributes, value 2.0
    masked_for_code = [
        lm if d == "code" else torch.zeros_like(lm)
        for d, lm in zip(domains, loss_masks)
    ]
    code_reducer = get_sum_of_sample_mean(
        total_lengths, response_lengths, masked_for_code,
        calculate_per_token_loss=False, qkv_format="thd", max_seq_lens=None,
    )
    result = code_reducer(x).item()
    assert result == 2.0, f"expected 2.0, got {result}"


def test_domain_reducer_returns_zero_for_absent_domain(monkeypatch):
    """When no sample matches the target domain, the reducer must return 0
    (not NaN, not error). aggregate_train_losses requires every microbatch
    emit the same key set — a domain with zero samples in this microbatch
    must contribute 0 to the positional aggregation."""
    _patch_parallel_state(monkeypatch)

    domains = ["math", "math"]  # no "code" samples
    response_lengths = [3, 3]
    total_lengths = [3, 3]
    loss_masks = [torch.ones(3), torch.ones(3)]
    x = torch.tensor([5.0, 5.0, 5.0, 7.0, 7.0, 7.0])

    masked_for_code = [torch.zeros_like(lm) for lm in loss_masks]  # all zero
    code_reducer = get_sum_of_sample_mean(
        total_lengths, response_lengths, masked_for_code,
        calculate_per_token_loss=False, qkv_format="thd", max_seq_lens=None,
    )
    result = code_reducer(x).item()
    assert result == 0.0, f"expected 0 for absent domain, got {result}"
    assert not torch.isnan(torch.tensor(result))


def test_per_domain_reductions_sum_to_global(monkeypatch):
    """When domains partition the batch and we use sum-mode (sum_of_sample_mean
    sums per-sample means), summing per-domain reductions equals the global one."""
    _patch_parallel_state(monkeypatch)

    domains = ["math", "code", "math", "code"]
    response_lengths = [2, 2, 2, 2]
    total_lengths = [2, 2, 2, 2]
    loss_masks = [torch.ones(2) for _ in range(4)]
    # per-sample means: [1, 2, 3, 4]
    x = torch.tensor([1.0]*2 + [2.0]*2 + [3.0]*2 + [4.0]*2)

    global_reducer = get_sum_of_sample_mean(
        total_lengths, response_lengths, loss_masks,
        calculate_per_token_loss=False, qkv_format="thd", max_seq_lens=None,
    )
    global_value = global_reducer(x).item()
    assert global_value == 1.0 + 2.0 + 3.0 + 4.0  # = 10.0

    # Sum per-domain reductions. {math}=samples 0,2 -> 1+3=4; {code}=1,3 -> 2+4=6.
    per_domain = 0.0
    for target in ["math", "code"]:
        masked = [lm if d == target else torch.zeros_like(lm) for d, lm in zip(domains, loss_masks)]
        red = get_sum_of_sample_mean(
            total_lengths, response_lengths, masked,
            calculate_per_token_loss=False, qkv_format="thd", max_seq_lens=None,
        )
        per_domain += red(x).item()

    assert per_domain == global_value, f"per-domain sum {per_domain} != global {global_value}"


# ---------------------------------------------------------------------------
# 2. Unified correctness signal: scalar fallback + metadata override
# ---------------------------------------------------------------------------
#
# The actual `_correctness` helper lives in miles.ray.rollout but importing
# that module pulls in ray. Vendor the implementation here as a fixture and
# verify the *semantic* contract — the implementation in rollout.py is a
# verbatim copy of this snippet (kept in sync by code review).


def _vendored_correctness(sample, args) -> bool:
    """Mirror of miles.ray.rollout._correctness — keep in sync."""
    if "correctness_reward" in sample.metadata:
        return sample.metadata["correctness_reward"] > 0
    val = sample.get_reward_value(args)
    return isinstance(val, (int, float)) and val > 0


@dataclass
class _FakeSample:
    reward: Any
    metadata: dict = field(default_factory=dict)

    def get_reward_value(self, args):
        return self.reward if not args.reward_key else self.reward[args.reward_key]


def test_correctness_scalar_fallback():
    args = Namespace(reward_key=None)
    assert _vendored_correctness(_FakeSample(1.0), args) is True
    assert _vendored_correctness(_FakeSample(0.5), args) is True
    assert _vendored_correctness(_FakeSample(0.0), args) is False
    assert _vendored_correctness(_FakeSample(-0.3), args) is False


def test_correctness_metadata_override_takes_precedence():
    """metadata['correctness_reward'] wins even when reward is also scalar."""
    args = Namespace(reward_key=None)
    s = _FakeSample(1.0, metadata={"correctness_reward": 0.0})  # scalar says correct, metadata says wrong
    assert _vendored_correctness(s, args) is False
    s = _FakeSample(0.0, metadata={"correctness_reward": 1.0})
    assert _vendored_correctness(s, args) is True


def test_correctness_non_scalar_reward_with_metadata():
    """OPD path: reward is a dict, correctness comes from metadata."""
    args = Namespace(reward_key=None)
    s = _FakeSample({"kl_a": 0.5, "kl_b": 0.3}, metadata={"correctness_reward": 1.0})
    assert _vendored_correctness(s, args) is True
    s = _FakeSample({"kl_a": 0.5}, metadata={"correctness_reward": 0.0})
    assert _vendored_correctness(s, args) is False


def test_correctness_non_scalar_reward_no_metadata_returns_false():
    """Without correctness_reward metadata and a non-scalar reward, the
    helper returns False (val>0 short-circuits via isinstance check)."""
    args = Namespace(reward_key=None)
    s = _FakeSample({"kl_a": 0.5})
    assert _vendored_correctness(s, args) is False


# ---------------------------------------------------------------------------
# 3. compute_samples_seen
# ---------------------------------------------------------------------------

def test_compute_samples_seen_first_rollout():
    args = Namespace(rollout_batch_size=8, n_samples_per_prompt=4)
    # rollout_id=0 means the first rollout has finished -> 32 samples seen.
    assert compute_samples_seen(args, 0) == 32


def test_compute_samples_seen_monotone():
    args = Namespace(rollout_batch_size=8, n_samples_per_prompt=4)
    seen = [compute_samples_seen(args, i) for i in range(5)]
    # Strictly monotone, increment of 32 per rollout.
    assert seen == [32, 64, 96, 128, 160]


# ---------------------------------------------------------------------------
# 4. Activation-by-presence: domains list is sorted-unique (matches the
#    DataIterator._all_domains_cache construction).
# ---------------------------------------------------------------------------

def test_all_domains_cache_construction():
    """Mirror the construction in get_batch: sorted({d for d in domains if d})."""
    domains = ["math", "code", None, "math", "code", "science", None]
    all_domains = sorted({d for d in domains if d})
    assert all_domains == ["code", "math", "science"]
    # None values are filtered out (samples without a domain don't add a key).
    assert None not in all_domains
