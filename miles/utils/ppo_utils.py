"""Compatibility export for the OPD reward helper introduced on stable."""

from miles.backends.training_utils.loss_hub.math_utils import compute_opd_reward

__all__ = ["compute_opd_reward"]
