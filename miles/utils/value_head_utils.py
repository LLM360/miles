"""Categorical / distributional value-head targets and decoding.

Implements HL-Gauss (Farebrother et al., "Stop Regressing: Training Value
Functions via Classification for Scalable Deep RL", 2024) and the one-hot /
two-hot / Bernoulli controls from "Start Classifying: Categorical Critics for
LLM Reinforcement Learning" (arXiv:2608.02181). The scalar clipped-MSE critic
(``--value-loss-type mse``, the default) is unaffected; these are opt-in.

The actor and GAE are unchanged: a categorical head is always decoded to a
scalar expectation ``E[V] = sum_k softmax(logits)_k * z_k`` before it is used
as ``V_old`` for advantages/returns. Only the critic's training loss becomes
cross-entropy against a target distribution built from ``returns``, instead
of squared error.

``--value-support-endpoints`` selects how bin centers are placed on
``[value_min, value_max]``: ``midpoint`` (default) matches the HL-Gauss
papers and never lands exactly on the endpoints; ``inclusive`` matches
C51/MuZero/DreamerV3 and can decode to exactly ``value_min``/``value_max``.
See :func:`build_value_support`.
"""

from __future__ import annotations

import math
from argparse import Namespace

import torch

_EPS = 1e-8


def value_head_output_size(args: Namespace) -> int:
    """Number of output units the value head needs for ``args.value_loss_type``."""
    value_loss_type = getattr(args, "value_loss_type", "mse")
    if value_loss_type == "mse":
        return 1
    if value_loss_type == "bernoulli":
        return 2
    return args.value_num_bins


def build_value_support(
    v_min: float,
    v_max: float,
    num_bins: int,
    endpoints: str = "midpoint",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(centers[K], edges[K+1])`` for a value support spanning ``[v_min, v_max]``.

    ``endpoints="midpoint"`` (default; Imani & White 2018 / HL-Gauss): ``K``
    equal-width bins, centers at each bin's middle. Centers never exactly
    equal ``v_min``/``v_max``; the resulting decode bias shrinks as
    ``1 / (2K)``.

    ``endpoints="inclusive"`` (C51 / MuZero two-hot / DreamerV3): ``K`` atoms
    placed at ``linspace(v_min, v_max, K)``, including the exact endpoints.
    Edges are midpoints between adjacent atoms, with the two outer edges at
    ``+-inf`` so tail mass beyond ``v_min``/``v_max`` still collapses onto
    the endpoint atom instead of being lost.
    """
    if endpoints == "midpoint":
        edges = torch.linspace(v_min, v_max, num_bins + 1, dtype=torch.float32)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, edges
    if endpoints == "inclusive":
        centers = torch.linspace(v_min, v_max, num_bins, dtype=torch.float32)
        inner_edges = (centers[:-1] + centers[1:]) / 2
        edges = torch.cat(
            [
                centers.new_full((1,), float("-inf")),
                inner_edges,
                centers.new_full((1,), float("inf")),
            ]
        )
        return centers, edges
    raise ValueError(f"Unknown endpoints convention: {endpoints}")


def get_value_support(args: Namespace, device: torch.device | None = None) -> torch.Tensor | None:
    """Bin centers used to decode a categorical value head to a scalar.

    Returns ``None`` for the default scalar (``mse``) critic.
    """
    value_loss_type = getattr(args, "value_loss_type", "mse")
    if value_loss_type == "mse":
        return None
    if value_loss_type == "bernoulli":
        centers = torch.tensor([0.0, 1.0], dtype=torch.float32)
    else:
        endpoints = getattr(args, "value_support_endpoints", "midpoint")
        centers, _ = build_value_support(args.value_min, args.value_max, args.value_num_bins, endpoints=endpoints)
    return centers.to(device) if device is not None else centers


def decode_categorical_value(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Decode ``[..., K]`` logits to a scalar expectation.

    ``E[V] = sum_k softmax(logits)_k * z_k``, matching the HL-Gauss paper's
    decode step for GAE/PPO, which stays scalar and unchanged.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    return (probs * support.to(probs.device, dtype=probs.dtype)).sum(dim=-1)


def hl_gauss_target(returns: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """HL-Gauss target: ``Normal(mean=returns, std=sigma)`` mass per bin, renormalized.

    Args:
        returns: Target scalars, shape ``[...]``.
        edges: Bin edges, shape ``[K + 1]``.
        sigma: Gaussian std, typically a fraction of the bin width.

    Returns:
        Target distribution, shape ``[..., K]``, rows sum to 1.
    """
    y = returns.float().unsqueeze(-1)
    edges = edges.to(y.device, dtype=torch.float32)
    cdf = 0.5 * (1.0 + torch.erf((edges - y) / (sigma * math.sqrt(2.0))))
    probs = cdf[..., 1:] - cdf[..., :-1]
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(_EPS)


def twohot_target(returns: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Two-hot target: split unit mass between the two bins neighboring ``returns``.

    This is an exact (clamped) decomposition of the scalar target onto the
    support, not a success/failure encoding.
    """
    centers = centers.to(returns.device, dtype=torch.float32)
    num_bins = centers.numel()
    y = returns.float().clamp(min=centers[0].item(), max=centers[-1].item())
    idx_hi = torch.searchsorted(centers.contiguous(), y.contiguous()).clamp(1, num_bins - 1)
    idx_lo = idx_hi - 1
    z_lo = centers[idx_lo]
    z_hi = centers[idx_hi]
    w_hi = ((y - z_lo) / (z_hi - z_lo).clamp_min(_EPS)).clamp(0.0, 1.0)
    probs = torch.zeros(*returns.shape, num_bins, device=returns.device, dtype=torch.float32)
    probs.scatter_(-1, idx_lo.unsqueeze(-1), (1.0 - w_hi).unsqueeze(-1))
    probs.scatter_add_(-1, idx_hi.unsqueeze(-1), w_hi.unsqueeze(-1))
    return probs


def onehot_target(returns: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """One-hot target at the nearest bin center to ``returns`` (larger-head control)."""
    centers = centers.to(returns.device, dtype=torch.float32)
    y = returns.float().unsqueeze(-1)
    idx = (y - centers).abs().argmin(dim=-1)
    return torch.nn.functional.one_hot(idx, num_classes=centers.numel()).float()


def bernoulli_target(returns: torch.Tensor) -> torch.Tensor:
    """Two-class soft target ``[1 - y, y]`` for ``y = returns`` clamped to ``[0, 1]``.

    Binary-classification control: treats the return itself as the target
    success probability. Intended for terminal 0/1-reward RLVR where
    ``returns`` is already close to ``[0, 1]``. Equivalent to a single
    sigmoid on ``logits[..., 1] - logits[..., 0]``, not two independent
    fail/success sigmoids.
    """
    y = returns.float().clamp(0.0, 1.0)
    return torch.stack([1.0 - y, y], dim=-1)


def categorical_value_target(returns: torch.Tensor, args: Namespace) -> torch.Tensor:
    """Dispatch to the target-encoding selected by ``args.value_loss_type``.

    Returns a target distribution ``[..., K]`` matching
    ``value_head_output_size(args)``.
    """
    value_loss_type = args.value_loss_type
    if value_loss_type == "bernoulli":
        return bernoulli_target(returns)
    endpoints = getattr(args, "value_support_endpoints", "midpoint")
    centers, edges = build_value_support(args.value_min, args.value_max, args.value_num_bins, endpoints=endpoints)
    if value_loss_type == "hl_gauss":
        # Bin width from the actual center spacing: (v_max-v_min)/K for midpoint,
        # (v_max-v_min)/(K-1) for inclusive.
        bin_width = (centers[1] - centers[0]).item() if centers.numel() > 1 else (args.value_max - args.value_min)
        sigma = args.value_hl_gauss_sigma_ratio * bin_width
        return hl_gauss_target(returns, edges, sigma)
    if value_loss_type == "twohot":
        return twohot_target(returns, centers)
    if value_loss_type == "onehot":
        return onehot_target(returns, centers)
    raise ValueError(f"Unknown value_loss_type: {value_loss_type}")


def cross_entropy_with_soft_target(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Per-element cross-entropy: ``-sum_k target_probs_k * log_softmax(logits)_k``.

    Args:
        logits: Raw (pre-softmax) categorical value-head outputs, ``[..., K]``.
        target_probs: Target distribution, ``[..., K]``, e.g. from
            :func:`categorical_value_target`.

    Returns:
        Per-element loss, ``[...]`` (last dim reduced away).
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(target_probs.to(log_probs.device, dtype=log_probs.dtype) * log_probs).sum(dim=-1)
