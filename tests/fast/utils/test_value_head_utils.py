from argparse import Namespace

import pytest
import torch

from miles.utils.value_head_utils import (
    bernoulli_target,
    build_value_support,
    categorical_value_target,
    cross_entropy_with_soft_target,
    decode_categorical_value,
    get_value_support,
    hl_gauss_target,
    onehot_target,
    twohot_target,
    value_head_output_size,
)


def _mse_args() -> Namespace:
    return Namespace(value_loss_type="mse")


def _categorical_args(value_loss_type: str, num_bins: int = 5, endpoints: str = "midpoint") -> Namespace:
    return Namespace(
        value_loss_type=value_loss_type,
        value_num_bins=num_bins,
        value_min=0.0,
        value_max=1.0,
        value_hl_gauss_sigma_ratio=0.75,
        value_support_endpoints=endpoints,
    )


def test_value_head_output_size_mse_is_scalar() -> None:
    assert value_head_output_size(_mse_args()) == 1


def test_value_head_output_size_bernoulli_is_two() -> None:
    assert value_head_output_size(_categorical_args("bernoulli")) == 2


@pytest.mark.parametrize("value_loss_type", ["hl_gauss", "twohot", "onehot"])
def test_value_head_output_size_matches_num_bins(value_loss_type: str) -> None:
    assert value_head_output_size(_categorical_args(value_loss_type, num_bins=51)) == 51


def test_get_value_support_is_none_for_mse() -> None:
    assert get_value_support(_mse_args()) is None


def test_get_value_support_bernoulli_is_zero_one() -> None:
    support = get_value_support(_categorical_args("bernoulli"))
    torch.testing.assert_close(support, torch.tensor([0.0, 1.0]))


def test_build_value_support_centers_and_edges() -> None:
    centers, edges = build_value_support(0.0, 1.0, num_bins=4)
    torch.testing.assert_close(edges, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    torch.testing.assert_close(centers, torch.tensor([0.125, 0.375, 0.625, 0.875]))


def test_build_value_support_inclusive_endpoints_include_min_max() -> None:
    centers, edges = build_value_support(0.0, 1.0, num_bins=5, endpoints="inclusive")
    torch.testing.assert_close(centers, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")
    torch.testing.assert_close(edges[1:-1], torch.tensor([0.125, 0.375, 0.625, 0.875]))


def test_build_value_support_rejects_unknown_endpoints() -> None:
    with pytest.raises(ValueError, match="Unknown endpoints convention"):
        build_value_support(0.0, 1.0, num_bins=5, endpoints="bogus")


@pytest.mark.parametrize("endpoints", ["midpoint", "inclusive"])
@pytest.mark.parametrize(
    "v_min,v_max",
    [(float("-inf"), 1.0), (0.0, float("inf")), (float("nan"), 1.0), (0.0, float("nan"))],
)
def test_build_value_support_rejects_nonfinite_bounds(endpoints: str, v_min: float, v_max: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        build_value_support(v_min, v_max, num_bins=5, endpoints=endpoints)


@pytest.mark.parametrize("sigma", [0.0, -0.1, float("inf"), float("-inf"), float("nan")])
def test_hl_gauss_rejects_invalid_sigma(sigma: float) -> None:
    _, edges = build_value_support(0.0, 1.0, num_bins=51)
    with pytest.raises(ValueError, match="finite and positive"):
        hl_gauss_target(torch.tensor([0.0, 0.5, 1.0]), edges, sigma=sigma)


@pytest.mark.parametrize("ratio", [0.0, -0.75, float("inf"), float("-inf"), float("nan")])
def test_categorical_hl_gauss_rejects_invalid_sigma_ratio(ratio: float) -> None:
    args = _categorical_args("hl_gauss", num_bins=51)
    args.value_hl_gauss_sigma_ratio = ratio
    with pytest.raises(ValueError, match="finite and positive"):
        categorical_value_target(torch.tensor([0.0, 1.0]), args)


@pytest.mark.parametrize("endpoints", ["midpoint", "inclusive"])
def test_hl_gauss_out_of_support_targets_remain_normalized_and_trainable(endpoints: str) -> None:
    args = _categorical_args("hl_gauss", num_bins=51, endpoints=endpoints)
    returns = torch.tensor([-100.0, -0.2, 1.2, 100.0])
    target = categorical_value_target(returns, args)
    assert torch.isfinite(target).all()
    assert (target >= 0).all()
    torch.testing.assert_close(target.sum(dim=-1), torch.ones_like(returns))
    assert target.argmax(dim=-1).tolist() == [0, 0, 50, 50]

    logits = torch.zeros_like(target, requires_grad=True)
    loss = cross_entropy_with_soft_target(logits, target)
    assert (loss > 0).all()
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum(dim=-1) > 0).all()


def test_hl_gauss_midpoint_clamps_returns_to_support_edges() -> None:
    args = _categorical_args("hl_gauss", num_bins=51)
    returns = torch.tensor([-0.2, 0.0, 0.3, 1.0, 1.2])
    actual = categorical_value_target(returns, args)
    expected = categorical_value_target(returns.clamp(0.0, 1.0), args)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("endpoints", ["midpoint", "inclusive"])
def test_hl_gauss_preserves_unclamped_cdf_formula_where_applicable(endpoints: str) -> None:
    _, edges = build_value_support(0.0, 1.0, num_bins=5, endpoints=endpoints)
    # Inclusive outer edges are infinite: out-of-support means retain their
    # original Gaussian tail mass, rather than being clamped to the atoms.
    returns = torch.tensor([-0.2, 0.0, 0.3, 1.0, 1.2] if endpoints == "inclusive" else [0.0, 0.3, 1.0])
    sigma = 0.15
    cdf = 0.5 * (1.0 + torch.erf((edges - returns.unsqueeze(-1)) / (sigma * 2.0**0.5)))
    mass = cdf[:, 1:] - cdf[:, :-1]
    expected = mass / mass.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(hl_gauss_target(returns, edges, sigma), expected)


def test_twohot_target_inclusive_endpoints_recovers_exact_min_max() -> None:
    centers, _ = build_value_support(0.0, 1.0, num_bins=5, endpoints="inclusive")
    target = twohot_target(torch.tensor([0.0, 1.0]), centers)
    expected = torch.zeros(2, 5)
    expected[0, 0] = 1.0
    expected[1, 4] = 1.0
    torch.testing.assert_close(target, expected)

    decoded = decode_categorical_value(torch.log(target.clamp_min(1e-12)), centers)
    torch.testing.assert_close(decoded, torch.tensor([0.0, 1.0]), atol=1e-4, rtol=1e-4)


def test_twohot_target_midpoint_endpoints_cannot_reach_exact_min_max() -> None:
    centers, _ = build_value_support(0.0, 1.0, num_bins=5, endpoints="midpoint")
    target = twohot_target(torch.tensor([0.0, 1.0]), centers)
    decoded = decode_categorical_value(torch.log(target.clamp_min(1e-12)), centers)
    assert decoded[0].item() > 0.0
    assert decoded[1].item() < 1.0


def test_hl_gauss_target_inclusive_endpoints_sums_to_one_no_nan() -> None:
    _, edges = build_value_support(0.0, 1.0, num_bins=5, endpoints="inclusive")
    returns = torch.tensor([0.0, 0.5, 1.0])
    target = hl_gauss_target(returns, edges, sigma=0.1)

    assert not torch.isnan(target).any()
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(3))


def test_categorical_value_target_respects_value_support_endpoints() -> None:
    inclusive_args = _categorical_args("twohot", num_bins=5, endpoints="inclusive")
    target = categorical_value_target(torch.tensor([1.0]), inclusive_args)
    expected = torch.zeros(1, 5)
    expected[0, 4] = 1.0
    torch.testing.assert_close(target, expected)


def test_hl_gauss_target_sums_to_one_and_peaks_near_mean() -> None:
    _, edges = build_value_support(0.0, 1.0, num_bins=5)
    returns = torch.tensor([0.5])
    target = hl_gauss_target(returns, edges, sigma=0.1)

    torch.testing.assert_close(target.sum(dim=-1), torch.ones(1))
    assert torch.argmax(target[0]).item() == 2  # bin covering [0.4, 0.6)


def test_twohot_target_splits_mass_between_neighbors() -> None:
    centers, _ = build_value_support(0.0, 1.0, num_bins=4)  # centers: .125 .375 .625 .875
    returns = torch.tensor([0.5])

    target = twohot_target(returns, centers)

    torch.testing.assert_close(target.sum(dim=-1), torch.ones(1))
    assert torch.allclose(target[0, 0], torch.tensor(0.0))
    assert torch.allclose(target[0, 3], torch.tensor(0.0))
    assert target[0, 1] > 0
    assert target[0, 2] > 0
    decoded = decode_categorical_value(torch.log(target.clamp_min(1e-12)), centers)
    torch.testing.assert_close(decoded, returns, atol=1e-5, rtol=1e-4)


def test_twohot_target_exact_on_bin_center_is_one_hot() -> None:
    centers, _ = build_value_support(0.0, 1.0, num_bins=4)
    target = twohot_target(centers[1:2], centers)
    expected = torch.zeros(1, 4)
    expected[0, 1] = 1.0
    torch.testing.assert_close(target, expected)


def test_twohot_target_clamps_out_of_range_returns() -> None:
    centers, _ = build_value_support(0.0, 1.0, num_bins=4)
    target = twohot_target(torch.tensor([-5.0, 5.0]), centers)
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(2))
    assert target[0, 0] == 1.0
    assert target[1, 3] == 1.0


def test_onehot_target_picks_nearest_bin() -> None:
    centers, _ = build_value_support(0.0, 1.0, num_bins=4)  # centers: .125 .375 .625 .875
    target = onehot_target(torch.tensor([0.6]), centers)
    expected = torch.zeros(1, 4)
    expected[0, 2] = 1.0  # nearest center to 0.6 is 0.625
    torch.testing.assert_close(target, expected)


def test_bernoulli_target_is_soft_success_probability() -> None:
    target = bernoulli_target(torch.tensor([0.0, 0.3, 1.0]))
    expected = torch.tensor([[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]])
    torch.testing.assert_close(target, expected)


def test_bernoulli_target_clamps_outside_zero_one() -> None:
    target = bernoulli_target(torch.tensor([-1.0, 2.0]))
    torch.testing.assert_close(target, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_decode_categorical_value_matches_expectation() -> None:
    support = torch.tensor([0.0, 1.0, 2.0])
    logits = torch.tensor([[100.0, 0.0, 0.0]])  # ~one-hot on bin 0
    decoded = decode_categorical_value(logits, support)
    torch.testing.assert_close(decoded, torch.tensor([0.0]), atol=1e-4, rtol=1e-4)


def test_categorical_value_target_dispatches_on_loss_type() -> None:
    returns = torch.tensor([0.5])
    for value_loss_type in ["hl_gauss", "twohot", "onehot"]:
        target = categorical_value_target(returns, _categorical_args(value_loss_type))
        assert target.shape == (1, 5)
        torch.testing.assert_close(target.sum(dim=-1), torch.ones(1))

    target = categorical_value_target(returns, _categorical_args("bernoulli"))
    torch.testing.assert_close(target, torch.tensor([[0.5, 0.5]]))


def test_categorical_value_target_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown value_loss_type"):
        categorical_value_target(torch.tensor([0.5]), _categorical_args("bogus"))


def test_cross_entropy_with_soft_target_is_zero_for_matching_one_hot() -> None:
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    loss = cross_entropy_with_soft_target(logits, target)
    assert loss.item() < 1e-3


def test_cross_entropy_with_soft_target_penalizes_mismatch() -> None:
    logits = torch.tensor([[10.0, -10.0]])
    target_match = torch.tensor([[1.0, 0.0]])
    target_mismatch = torch.tensor([[0.0, 1.0]])

    loss_match = cross_entropy_with_soft_target(logits, target_match)
    loss_mismatch = cross_entropy_with_soft_target(logits, target_mismatch)

    assert loss_match.item() < loss_mismatch.item()


def test_hl_gauss_round_trip_decode_close_to_return_for_narrow_sigma() -> None:
    centers, edges = build_value_support(0.0, 1.0, num_bins=51)
    returns = torch.tensor([0.3, 0.7])
    target = hl_gauss_target(returns, edges, sigma=0.01)
    decoded = decode_categorical_value(torch.log(target.clamp_min(1e-12)), centers)
    torch.testing.assert_close(decoded, returns, atol=1e-2, rtol=1e-2)
