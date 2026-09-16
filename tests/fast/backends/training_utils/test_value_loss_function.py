"""`value_loss_function` branch coverage: default clipped-MSE vs categorical CE.

Uses a CP=1, `qkv_format="thd"` synthetic batch (no real megatron/distributed
init needed) to exercise `get_values`/`get_value_logits`/`value_loss_function`
end to end, for both the unchanged scalar (`mse`) path and the new
categorical (`hl_gauss`/`twohot`/`onehot`/`bernoulli`) paths from
arXiv:2608.02181.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.cp_utils import _empty_response_like, _pad_token_chunk, get_sum_of_sample_mean
from miles.backends.training_utils.loss import get_value_logits, get_values, value_loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.value_head_utils import categorical_value_target


@pytest.fixture(autouse=True)
def _cp1_parallel_state():
    group_info = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(intra_dp=group_info, intra_dp_cp=group_info, cp=group_info, tp=group_info)
    )
    yield


def _base_args(**overrides) -> Namespace:
    defaults = dict(
        qkv_format="thd",
        allgather_cp=False,
        rollout_temperature=1.0,
        calculate_per_token_loss=False,
        loss_agg_mode=None,
        value_clip=0.2,
        share_backbone_critic=False,
        value_loss_type="mse",
        value_hl_gauss_sigma_ratio=0.75,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_batch(returns: list[torch.Tensor], response_lengths: list[int], total_lengths: list[int]):
    loss_masks = [torch.ones(r) for r in response_lengths]
    unconcat_tokens = [torch.zeros(t, dtype=torch.long) for t in total_lengths]
    return {
        "returns": returns,
        "loss_masks": loss_masks,
        "unconcat_tokens": unconcat_tokens,
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }


def _sum_of_sample_mean(batch, args):
    return get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        loss_agg_mode=args.loss_agg_mode,
    )


# response_lengths=[2, 3], total_lengths=[3, 4] -> T = 7 packed tokens.
_RESPONSE_LENGTHS = [2, 3]
_TOTAL_LENGTHS = [3, 4]
_NUM_RESPONSE_TOKENS = sum(_RESPONSE_LENGTHS)


def test_mse_value_loss_matches_hand_computed_clipped_mse() -> None:
    args = _base_args(value_loss_type="mse")
    batch = _make_batch(
        returns=[torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])],
        response_lengths=_RESPONSE_LENGTHS,
        total_lengths=_TOTAL_LENGTHS,
    )
    batch["values"] = [torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])]  # V_old, all 0.5

    logits = torch.zeros(1, sum(_TOTAL_LENGTHS), 1)
    logits[0, [0, 1, 3, 4, 5], 0] = 0.9  # response-token positions (end-1) predict 0.9

    loss, log = value_loss_function(args, batch, logits, _sum_of_sample_mean(batch, args))

    # V_old=0.5, V_new=0.9 for every response token; clip=0.2 -> clipped to 0.7.
    # surr1=(0.7-returns)^2, surr2=(0.9-returns)^2; loss=max per token.
    expected = torch.tensor(
        [
            max((0.7 - 1.0) ** 2, (0.9 - 1.0) ** 2),
            max((0.7 - 1.0) ** 2, (0.9 - 1.0) ** 2),
            max((0.7 - 0.0) ** 2, (0.9 - 0.0) ** 2),
            max((0.7 - 0.0) ** 2, (0.9 - 0.0) ** 2),
            max((0.7 - 0.0) ** 2, (0.9 - 0.0) ** 2),
        ]
    )
    # sample-mean reducer: mean within each sample, summed across samples.
    expected_loss = expected[:2].mean() + expected[2:].mean()

    torch.testing.assert_close(loss, expected_loss, atol=1e-5, rtol=1e-4)
    # Every response token is clipped; sample-mean reducer sums per-sample means (1.0 + 1.0).
    assert log["value_clipfrac"].item() == pytest.approx(2.0)


@pytest.mark.parametrize("value_loss_type", ["hl_gauss", "twohot", "onehot"])
def test_categorical_value_loss_is_minimized_by_logits_matching_target(value_loss_type: str) -> None:
    """CE(target, softmax(logits)) >= H(target), with equality iff softmax(logits) == target
    (Gibbs' inequality). So feeding back `logits = log(target)` must give the lowest possible
    loss for that batch, strictly below a uniform (zero-logit) prediction — regardless of how
    each value_loss_type's target happens to be shaped (e.g. hl_gauss spreads mass across
    several neighboring bins, so a literal one-hot prediction is *not* the best case there).
    """
    args = _base_args(value_loss_type=value_loss_type, value_num_bins=5, value_min=0.0, value_max=1.0)
    returns = [torch.tensor([0.1, 0.9]), torch.tensor([0.5, 0.5, 0.5])]
    batch = _make_batch(returns=returns, response_lengths=_RESPONSE_LENGTHS, total_lengths=_TOTAL_LENGTHS)
    reducer = _sum_of_sample_mean(batch, args)

    num_bins = 5
    uniform_logits = torch.zeros(1, sum(_TOTAL_LENGTHS), num_bins, requires_grad=True)
    loss_uniform, _ = value_loss_function(args, batch, uniform_logits, reducer)

    target = categorical_value_target(torch.cat(returns), args)  # [5, num_bins]
    matching_logits = torch.zeros(1, sum(_TOTAL_LENGTHS), num_bins, requires_grad=True)
    with torch.no_grad():
        for pos, target_row in zip([0, 1, 3, 4, 5], target):
            # target may have exact 0.0 entries (twohot/onehot); clamp before log to avoid
            # -inf logits, which would make 0 * log_softmax(-inf) evaluate to nan below.
            matching_logits[0, pos] = target_row.clamp_min(1e-6).log()
    loss_matching, log_matching = value_loss_function(args, batch, matching_logits, reducer)

    assert loss_matching.item() < loss_uniform.item()
    assert log_matching["value_clipfrac"].item() == 0.0  # no clip for categorical heads

    loss_matching.backward()
    assert matching_logits.grad is not None
    assert torch.any(matching_logits.grad != 0)


def test_bernoulli_value_loss_matches_manual_cross_entropy() -> None:
    args = _base_args(value_loss_type="bernoulli")
    returns = [torch.tensor([1.0, 0.0]), torch.tensor([0.5, 0.5, 0.5])]
    batch = _make_batch(returns=returns, response_lengths=_RESPONSE_LENGTHS, total_lengths=_TOTAL_LENGTHS)
    reducer = _sum_of_sample_mean(batch, args)

    logits = torch.zeros(1, sum(_TOTAL_LENGTHS), 2)
    logits[0, [0, 1, 3, 4, 5], 1] = 2.0  # some logit mass on class-1 for every response token

    loss, log = value_loss_function(args, batch, logits, reducer)

    log_probs = torch.log_softmax(torch.tensor([0.0, 2.0]), dim=-1)
    ce_success = -log_probs[1].item()  # target [0, 1] (y=1.0)
    ce_fail = -log_probs[0].item()  # target [1, 0] (y=0.0)
    ce_half = -(0.5 * log_probs[0].item() + 0.5 * log_probs[1].item())  # target [0.5, 0.5]

    expected_loss = (ce_success + ce_fail) / 2 + ce_half  # sample-mean per sample, summed
    torch.testing.assert_close(loss.item(), expected_loss, atol=1e-5, rtol=1e-4)
    assert log["value_clipfrac"].item() == 0.0


def test_get_values_decodes_categorical_head_for_gae() -> None:
    args = _base_args(value_loss_type="hl_gauss", value_num_bins=5, value_min=0.0, value_max=1.0)
    unconcat_tokens = [torch.zeros(t, dtype=torch.long) for t in _TOTAL_LENGTHS]

    logits = torch.zeros(1, sum(_TOTAL_LENGTHS), 5)
    logits[0, [0, 1, 3, 4, 5], 4] = 10.0  # near value_max for every response token

    result = get_values(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=_TOTAL_LENGTHS,
        response_lengths=_RESPONSE_LENGTHS,
    )
    values = torch.cat([v.flatten() for v in result["values"]])
    assert values.shape == (_NUM_RESPONSE_TOKENS,)
    assert torch.all(values > 0.85)  # decoded near the top bin center, not the raw logit


def test_get_values_does_not_apply_temperature_to_categorical_head() -> None:
    args = _base_args(
        value_loss_type="hl_gauss", value_num_bins=5, value_min=0.0, value_max=1.0, rollout_temperature=0.5
    )
    unconcat_tokens = [torch.zeros(t, dtype=torch.long) for t in _TOTAL_LENGTHS]
    logits = torch.zeros(1, sum(_TOTAL_LENGTHS), 5)
    logits[0, [0, 1, 3, 4, 5], 4] = 10.0

    result_temp_on = get_values(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=_TOTAL_LENGTHS,
        response_lengths=_RESPONSE_LENGTHS,
        apply_temperature=True,
    )
    result_temp_off = get_values(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=_TOTAL_LENGTHS,
        response_lengths=_RESPONSE_LENGTHS,
        apply_temperature=False,
    )
    # Categorical head: apply_temperature is forced off regardless of the caller's flag.
    for a, b in zip(result_temp_on["values"], result_temp_off["values"], strict=False):
        torch.testing.assert_close(a, b)


def test_get_value_logits_returns_raw_bins_without_decode() -> None:
    args = _base_args(value_loss_type="hl_gauss", value_num_bins=5, value_min=0.0, value_max=1.0)
    unconcat_tokens = [torch.zeros(t, dtype=torch.long) for t in _TOTAL_LENGTHS]
    logits = torch.arange(sum(_TOTAL_LENGTHS) * 5, dtype=torch.float32).reshape(1, sum(_TOTAL_LENGTHS), 5)

    chunks = get_value_logits(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=_TOTAL_LENGTHS,
        response_lengths=_RESPONSE_LENGTHS,
    )
    assert [c.shape for c in chunks] == [(2, 5), (3, 5)]


def test_pad_token_chunk_1d_matches_legacy_last_dim_pad() -> None:
    value = torch.tensor([1.0, 2.0])
    padded = _pad_token_chunk(value, resp_start=1, response_length=5)
    torch.testing.assert_close(padded, torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0]))


def test_pad_token_chunk_2d_pads_tokens_not_bins() -> None:
    value = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    padded = _pad_token_chunk(value, resp_start=1, response_length=4)
    assert padded.shape == (4, 3)
    torch.testing.assert_close(padded[1:3], value)
    torch.testing.assert_close(padded[0], torch.zeros(3))
    torch.testing.assert_close(padded[3], torch.zeros(3))


def test_empty_response_like_keeps_trailing_dims() -> None:
    value = torch.zeros(0, 5, requires_grad=True)
    empty = _empty_response_like(value, response_length=3)
    assert empty.shape == (3, 5)
    assert empty.requires_grad
