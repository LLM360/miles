import os
from argparse import Namespace
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from miles.backends.training_utils.loss import get_log_probs_and_entropy
from miles.utils.types import RolloutSamplingMask


def verify_top_p_replay() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 4

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    cp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    cp_rank, tp_rank = divmod(rank, 2)
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(size=2, rank=tp_rank, group=tp_groups[cp_rank]),
        cp=SimpleNamespace(size=2, rank=cp_rank, group=cp_groups[tp_rank]),
    )

    full_logits = torch.randn(16, 8, generator=torch.Generator().manual_seed(7)).to(device).requires_grad_()
    tokens = [torch.arange(8, dtype=torch.long, device=device), torch.tensor([0, 1, 2, 3, 4, 5, 6, 3], device=device)]
    supports = [
        [[0, 2, 5], [3, 4], [1, 4, 7], [0, 5], [2, 6, 7], [1, 3, 7]],
        [[0, 3]],
    ]
    # Two independently packed sequences. The second response's sole logit is
    # position 6, so CP rank 1 has no response rows for that sample.
    cp_positions = [0, 1, 6, 7, 8, 9, 14, 15] if cp_rank == 0 else [2, 3, 4, 5, 10, 11, 12, 13]
    vocab_start = tp_rank * 4
    local_logits = (
        full_logits[cp_positions, vocab_start : vocab_start + 4].unsqueeze(0).contiguous().detach().requires_grad_()
    )
    args = Namespace(
        qkv_format="thd",
        rollout_temperature=0.7,
        allgather_cp=False,
        log_probs_chunk_size=1,
        true_on_policy_mode=False,
    )

    saved_dense_masks = []

    def pack(tensor):
        if tensor.dtype == torch.bool and tensor.ndim == 2 and tensor.size(-1) == local_logits.size(-1):
            saved_dense_masks.append(tuple(tensor.shape))
        return tensor

    # Also cover singleton observations and support wholly owned by the other TP rank.
    singleton_supports = [[[2], [0, 3], [4, 7], [5], [6], [7]], [[3]]]
    for rows, chunk_size in product((None, supports, singleton_supports), (-1, 1, 3)):
        sampling_masks = [RolloutSamplingMask.from_rows(sample_rows) for sample_rows in rows] if rows else None
        args.log_probs_chunk_size = chunk_size
        saved_dense_masks.clear()

        with (
            torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
            patch("miles.backends.training_utils.loss.get_parallel_state", return_value=parallel_state),
            patch("miles.backends.training_utils.cp_utils.get_parallel_state", return_value=parallel_state),
        ):
            result = get_log_probs_and_entropy(
                local_logits,
                args=args,
                unconcat_tokens=tokens,
                total_lengths=[8, 8],
                response_lengths=[6, 1],
                with_entropy=True,
                rollout_sampling_masks=sampling_masks,
            )

        assert not saved_dense_masks, f"dense sampling masks retained for backward: {saved_dense_masks}"

        owned_rows = [[0, 5], [0]] if cp_rank == 0 else [[1, 2, 3, 4], []]
        scaled_logits = full_logits / args.rollout_temperature
        expected = {"log_probs": [], "entropy": []}
        for sample_idx, (response_length, logit_start) in enumerate(((6, 1), (1, 14))):
            log_probs, entropy = [], []
            for response_row in owned_rows[sample_idx]:
                support = torch.tensor(
                    rows[sample_idx][response_row] if rows is not None else range(full_logits.size(-1)),
                    dtype=torch.long,
                    device=device,
                )
                full_row = scaled_logits[logit_start + response_row]
                selected = torch.nonzero(support == tokens[sample_idx][-response_length + response_row]).item()
                log_probs.append(torch.log_softmax(full_row[support], dim=-1)[selected])
                entropy.append(-(torch.softmax(full_row, dim=-1) * torch.log_softmax(full_row, dim=-1)).sum())

            for key, values in (("log_probs", log_probs), ("entropy", entropy)):
                expected[key].append(torch.stack(values) if values else full_logits.new_empty(0))
                torch.testing.assert_close(result[key][sample_idx], expected[key][-1], atol=1e-5, rtol=1e-5)

        for key in expected:
            actual_gradient = torch.autograd.grad(torch.cat(result[key]).sum(), local_logits, retain_graph=True)[0]
            full_gradient = torch.autograd.grad(torch.cat(expected[key]).sum(), full_logits, retain_graph=True)[0]
            expected_gradient = full_gradient[cp_positions, vocab_start : vocab_start + 4].unsqueeze(0)
            # The existing fused CE returns BF16 gradients; entropy gradients stay FP32.
            tolerance = 4e-3 if key == "log_probs" else 1e-5
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance)
            if key == "log_probs":
                assert torch.count_nonzero(actual_gradient[expected_gradient == 0]).item() == 0

    dist.barrier()
    if rank == 0:
        print("TP2 x CP2 top-p replay PASS")


def main() -> None:
    try:
        verify_top_p_replay()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
