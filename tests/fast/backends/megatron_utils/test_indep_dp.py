import os
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.megatron_utils.indep_dp import _intra_cell_consensus, _zero_grad_buffers


def _init_gloo(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _run(fn: Any, world_size: int = 2) -> None:
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


# -- _intra_cell_consensus tests (multi-process with native Gloo PG) --


def _worker_consensus_all_true(rank: int, world_size: int) -> None:
    _init_gloo(rank, world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        result = _intra_cell_consensus(success=True, gloo_group=group)
        assert result is True, f"rank {rank}: expected True, got {result}"
    finally:
        dist.destroy_process_group()


def test_intra_cell_consensus_all_true() -> None:
    _run(_worker_consensus_all_true)


def _worker_consensus_all_false(rank: int, world_size: int) -> None:
    _init_gloo(rank, world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        result = _intra_cell_consensus(success=False, gloo_group=group)
        assert result is False, f"rank {rank}: expected False, got {result}"
    finally:
        dist.destroy_process_group()


def test_intra_cell_consensus_all_false() -> None:
    _run(_worker_consensus_all_false)


def _worker_consensus_mixed(rank: int, world_size: int) -> None:
    _init_gloo(rank, world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        success = rank == 0  # rank 0: True, rank 1: False
        result = _intra_cell_consensus(success=success, gloo_group=group)
        assert result is False, f"rank {rank}: expected False, got {result}"
    finally:
        dist.destroy_process_group()


def test_intra_cell_consensus_mixed() -> None:
    _run(_worker_consensus_mixed)


# -- _zero_grad_buffers tests (single process, mock model) --


def _make_mock_bucket(shape: tuple[int, ...] = (4,)) -> MagicMock:
    bucket = MagicMock()
    bucket.grad_data = torch.ones(shape)
    return bucket


def _make_mock_bucket_group(num_buckets: int = 2) -> MagicMock:
    bucket_group = MagicMock()
    bucket_group.buckets = [_make_mock_bucket() for _ in range(num_buckets)]
    return bucket_group


def _make_mock_model_chunk(
    num_bucket_groups: int = 1,
    num_expert_bucket_groups: int = 1,
) -> MagicMock:
    chunk = MagicMock()
    chunk.bucket_groups = [_make_mock_bucket_group() for _ in range(num_bucket_groups)]
    chunk.expert_parallel_bucket_groups = [_make_mock_bucket_group() for _ in range(num_expert_bucket_groups)]
    return chunk


class TestZeroGradBuffers:
    def test_zeros_all_grad_data(self) -> None:
        model = [_make_mock_model_chunk(), _make_mock_model_chunk()]

        # Verify non-zero before
        for chunk in model:
            for bg in chunk.bucket_groups + chunk.expert_parallel_bucket_groups:
                for bucket in bg.buckets:
                    assert bucket.grad_data.abs().sum().item() > 0

        _zero_grad_buffers(model)

        # Verify all zeros after
        for chunk in model:
            for bg in chunk.bucket_groups + chunk.expert_parallel_bucket_groups:
                for bucket in bg.buckets:
                    assert bucket.grad_data.abs().sum().item() == 0.0

    def test_empty_model(self) -> None:
        _zero_grad_buffers([])

    def test_empty_bucket_groups(self) -> None:
        chunk = MagicMock()
        chunk.bucket_groups = []
        chunk.expert_parallel_bucket_groups = []
        _zero_grad_buffers([chunk])
