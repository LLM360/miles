from unittest.mock import patch

from miles.utils import wandb_utils


def test_top_level_metrics_use_their_logical_step_axes() -> None:
    expected = {
        "optimization/*": "train/step",
        "policy_shift/*": "train/step",
        "train_inference_mismatch/*": "train/step",
        "policy_shift/ref_log_probs": "rollout/step",
        "train_inference_mismatch/log_probs": "rollout/step",
        "train_inference_mismatch/rollout_log_probs": "rollout/step",
        "agent/*": "rollout/step",
    }

    with patch.object(wandb_utils.wandb, "define_metric") as define_metric:
        wandb_utils._init_wandb_common()

    defined = {
        call.args[0]: call.kwargs.get("step_metric")
        for call in define_metric.call_args_list
        if call.args[0] in expected
    }
    assert defined == expected
