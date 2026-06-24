import math

import torch

from miles.backends.training_utils.loss import get_oapl_targets


def test_get_oapl_targets_uses_group_logmeanexp_baseline():
    targets = get_oapl_targets(
        rewards=[0.0, 1.0, 1.0, 0.0],
        group_indices=[10, 10, 11, 11],
        n_samples_per_prompt=2,
        beta1=1.0,
        device="cpu",
    )

    baseline = math.log((math.exp(0.0) + math.exp(1.0)) / 2.0)
    expected = torch.tensor(
        [
            0.0 - baseline,
            1.0 - baseline,
            1.0 - baseline,
            0.0 - baseline,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(torch.stack(targets), expected)


def test_get_oapl_targets_falls_back_to_contiguous_prompt_groups():
    targets = get_oapl_targets(
        rewards=[0.0, 1.0, 0.0, 0.0],
        group_indices=None,
        n_samples_per_prompt=2,
        beta1=0.5,
        device="cpu",
    )

    group0 = 0.5 * math.log((math.exp(0.0 / 0.5) + math.exp(1.0 / 0.5)) / 2.0)
    group1 = 0.5 * math.log((math.exp(0.0 / 0.5) + math.exp(0.0 / 0.5)) / 2.0)
    expected = torch.tensor(
        [
            0.0 - group0,
            1.0 - group0,
            0.0 - group1,
            0.0 - group1,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(torch.stack(targets), expected)
