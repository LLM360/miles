from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.model import train_one_step
from miles.backends.training_utils.data import DataIterator, get_rollout_data
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.rollout import RolloutManager
from miles.utils.types import RolloutSamplingMask, Sample


def test_replay_mask_reaches_scoring_and_training():
    samples = [
        Sample(
            index=i,
            tokens=[i + 1] + response,
            response_length=len(response),
            rollout_log_probs=[-0.1 * (i + 1)] * len(response),
            rollout_sampling_mask=RolloutSamplingMask.from_rows(support),
        )
        for i, (response, support) in enumerate(
            [([4], [[4]]), ([5], [[1, 5]]), ([7, 8], [[7, 9], [2, 8]]), ([9], [[0, 9]])]
        )
    ]
    args = Namespace(
        loss_type="policy_loss",
        rollout_top_p=0.9,
        balance_data=False,
        data_pad_size_multiplier=1,
        qkv_format="bshd",
        allgather_cp=False,
        use_rollout_entropy=False,
        use_rollout_logprobs=True,
        use_dynamic_batch_size=True,
        use_dynamic_global_batch_size=False,
        calculate_per_token_loss=False,
        recompute_loss_function=False,
        enable_mtp_training=False,
        fp16=False,
        bf16=False,
        dumper_enable=False,
        dumper_fwd_only=[],
        dumper_fwd_bwd=[],
        custom_megatron_before_log_prob_hook_path=None,
        custom_megatron_before_train_step_hook_path=None,
        seq_length=3,
        micro_batch_size=1,
        global_batch_size=4,
        decoder_seq_length=None,
        ci_test=False,
        use_opsm=False,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        get_mismatch_metrics=False,
        use_tis=False,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = args
    manager.custom_convert_samples_to_train_data_func = None
    manager._post_process_rewards = lambda _: ([0.0] * 4, [0.0] * 4)
    group = GroupInfo(rank=0, size=1, group=None)
    dp_group = GroupInfo(rank=0, size=2, group=None)
    parallel = ParallelState(intra_dp=dp_group, intra_dp_cp=dp_group, cp=group, tp=group)
    scored = []

    def forward(*, input_ids, **kwargs):
        return torch.zeros(*input_ids.shape, 10, requires_grad=True)

    def schedule(*, forward_step_func, data_iterator, model, num_microbatches, forward_only, **kwargs):
        results = []
        for _ in range(num_microbatches):
            output, callback = forward_step_func(data_iterator[0], model[0])
            result = callback(output)
            if not forward_only:
                loss, _, result = result
                loss.backward()
                assert output.grad is not None and output.grad.abs().sum() > 0
            results.append(result)
        return results

    def score(logits, *, unconcat_tokens, response_lengths, rollout_sampling_masks=None, **kwargs):
        assert len(unconcat_tokens) == 1
        sample = samples[unconcat_tokens[0][0].item() - 1]
        assert unconcat_tokens[0].tolist() == sample.tokens
        assert response_lengths == [sample.response_length]
        assert rollout_sampling_masks == [sample.rollout_sampling_mask]
        scored.append(sample.index)
        log_probs = logits[0, : sample.response_length, 0] - 0.1 * (sample.index + 1)
        return {"log_probs": [log_probs], "entropy": [log_probs * 0]}

    model = MagicMock(config=SimpleNamespace(), side_effect=forward)
    optimizer = MagicMock()
    optimizer.step.return_value = (True, 1.0, 0)
    with (
        patch("miles.backends.training_utils.parallel._parallel_state", parallel),
        patch("torch.cuda.current_device", return_value="cpu"),
        patch("miles.ray.rollout.ray.put", side_effect=lambda data: SimpleNamespace(data=data, hex=lambda: "test")),
        patch("miles.utils.data.ray.get", side_effect=lambda ref: ref.data),
        patch("miles.backends.megatron_utils.model.get_args", return_value=args),
        patch("miles.backends.megatron_utils.model.get_forward_backward_func", return_value=schedule),
        patch("miles.backends.megatron_utils.model.mpu.is_pipeline_last_stage", return_value=True),
        patch("miles.backends.megatron_utils.model.clear_memory"),
        patch("miles.backends.training_utils.log_utils.dist.all_reduce"),
        patch("miles.backends.megatron_utils.actor.get_log_probs_and_entropy", side_effect=score),
        patch("miles.backends.training_utils.loss.get_log_probs_and_entropy", side_effect=score),
    ):
        converted = manager._convert_samples_to_train_data(samples)
        shards = manager._split_train_data_by_dp(converted, dp_size=2)
        for rank, shard in enumerate(shards):
            assert shard.inner.data["sample_indices"] == [rank, rank + 2]
            assert shard.inner.data["rollout_sampling_masks"] == [
                samples[i].rollout_sampling_mask for i in (rank, rank + 2)
            ]
        data = get_rollout_data(args, shards)
        assert data["sample_indices"] == [0, 2]
        assert data["total_lengths"] == [2, 3]
        data["advantages"] = [torch.ones(length) for length in data["response_lengths"]]
        iterator = DataIterator(data, micro_batch_indices=[[1], [0]])
        actor = object.__new__(MegatronTrainRayActor)
        actor.args, actor.model = args, [model]
        for prefix in ("", "ref_"):
            scored.clear()
            result = actor.compute_log_prob([iterator], [2], store_prefix=prefix)
            assert scored == [2, 0]
            assert f"{prefix}log_probs" in result
            for actual, expected in zip(result[f"{prefix}log_probs"], data["rollout_log_probs"], strict=True):
                torch.testing.assert_close(actual, expected)
            data.update(result)

        scored.clear()
        train_one_step(args, 0, 0, [iterator.reset()], [model], optimizer, MagicMock(), 2)
        assert scored == [2, 0]
        optimizer.step.assert_called_once()
