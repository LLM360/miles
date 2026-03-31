"""Tests for process_group_utils: GroupInfo, GroupsInfo, GeneralPGUtil, MultiPGUtil."""

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from miles.utils.process_group_utils import (
    GeneralPGUtil,
    GroupInfo,
    GroupsInfo,
    MultiPGUtil,
    _check_wait,
    _NativePGUtil,
    _RawPGUtil,
)


def _init_gloo(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _run(fn, world_size: int = 4) -> None:
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


def _make_mesh():
    return init_device_mesh("cpu", mesh_shape=(2, 2), mesh_dim_names=("outer", "inner"))


# -- GroupInfo / GroupsInfo tests (no distributed needed) --


class TestGroupInfo:
    def test_construction_with_none_group(self) -> None:
        info = GroupInfo(rank=0, size=4, group=None)
        assert info.rank == 0
        assert info.size == 4
        assert info.gloo_group is None


class TestGroupsInfo:
    def test_from_single(self) -> None:
        info = GroupInfo(rank=2, size=4, group=None)
        result = GroupsInfo.from_single(info)
        assert result.rank == 2
        assert result.size == 4
        assert result.groups_inner_to_outer == [None]
        assert result.gloo_groups_inner_to_outer == [None]

    def test_from_single_with_gloo(self) -> None:
        sentinel_group = object()
        sentinel_gloo = object()
        info = GroupInfo(rank=0, size=2, group=sentinel_group, gloo_group=sentinel_gloo)
        result = GroupsInfo.from_single(info)
        assert result.groups_inner_to_outer == [sentinel_group]
        assert result.gloo_groups_inner_to_outer == [sentinel_gloo]

    def test_from_pair(self) -> None:
        inner = GroupInfo(rank=1, size=3, group=None)
        outer = GroupInfo(rank=2, size=4, group=None)
        result = GroupsInfo.from_pair(inner=inner, outer=outer)
        assert result.rank == 2 * 3 + 1  # 7
        assert result.size == 4 * 3  # 12
        assert result.gloo_groups_inner_to_outer == [None, None]

    def test_from_pair_with_gloo(self) -> None:
        inner_gloo = object()
        outer_gloo = object()
        inner = GroupInfo(rank=0, size=2, group=None, gloo_group=inner_gloo)
        outer = GroupInfo(rank=0, size=3, group=None, gloo_group=outer_gloo)
        result = GroupsInfo.from_pair(inner=inner, outer=outer)
        assert result.gloo_groups_inner_to_outer == [inner_gloo, outer_gloo]

    def test_from_pair_rank_zero_only_when_both_zero(self) -> None:
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert result.rank == 0
        assert result.size == 6

    def test_from_pair_rank_nonzero_when_inner_nonzero(self) -> None:
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=1, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert result.rank == 1

    def test_from_pair_rank_nonzero_when_outer_nonzero(self) -> None:
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=1, size=3, group=None),
        )
        assert result.rank == 2


# -- Parameterized GeneralPGUtil tests (native vs torchft code paths) --


UTIL_CLASSES = [_NativePGUtil, _RawPGUtil]


def _worker_pg_util_ops(rank: int, world_size: int) -> None:
    """Test all GeneralPGUtil operations with both native and torchft code paths."""
    _init_gloo(rank, world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        for util_cls in UTIL_CLASSES:
            util = util_cls()

            # get_rank / get_size
            assert util.get_rank(group) == rank
            assert util.get_size(group) == world_size

            # all_reduce SUM
            tensor = torch.tensor([float(rank + 1)])
            util.all_reduce(tensor, group, op=dist.ReduceOp.SUM)
            assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

            # reduce to root
            tensor = torch.tensor([float(rank + 1)])
            util.reduce(tensor, group, op=dist.ReduceOp.SUM)
            if rank == 0:
                assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

            # broadcast from root
            tensor = torch.tensor([99.0]) if rank == 0 else torch.tensor([0.0])
            util.broadcast(tensor, group)
            assert tensor.item() == 99.0

            # all_gather
            input_t = torch.tensor([float(rank)])
            output_t = [torch.zeros(1) for _ in range(world_size)]
            util.all_gather(output_t, input_t, group=group)
            assert [t.item() for t in output_t] == [0.0, 1.0, 2.0, 3.0]

            # gather
            input_t = torch.tensor([float(rank)])
            if rank == 0:
                gather_list = [torch.zeros(1) for _ in range(world_size)]
                util.gather(input_t, gather_list=gather_list, dst=0, group=group)
                assert [t.item() for t in gather_list] == [0.0, 1.0, 2.0, 3.0]
            else:
                util.gather(input_t, gather_list=None, dst=0, group=group)

        # GroupInfo verification
        GroupInfo(rank=rank, size=world_size, group=group)
        wrong_rank = (rank + 1) % world_size
        with pytest.raises(AssertionError):
            GroupInfo(rank=wrong_rank, size=world_size, group=group)
    finally:
        dist.destroy_process_group()


def test_pg_util_ops() -> None:
    _run(_worker_pg_util_ops)


def _worker_gather_object_native_vs_raw(rank: int, world_size: int) -> None:
    """Verify _NativePGUtil and _RawPGUtil gather_object produce identical results."""
    _init_gloo(rank, world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        test_objects = [
            {"rank": rank, "value": rank * 10},
            [rank, rank + 1, "hello"],
            f"string_from_rank_{rank}",
            (rank, {"nested": True}),
        ]

        def _gather(util: GeneralPGUtil, obj: Any) -> list[Any] | None:
            if rank == 0:
                result: list[Any] = [None] * world_size
                util.gather_object(obj, result, dst=0, group=group)
                return result
            else:
                util.gather_object(obj, None, dst=0, group=group)
                return None

        for obj in test_objects:
            native_result = _gather(_NativePGUtil(), obj)
            raw_result = _gather(_RawPGUtil(), obj)
            if rank == 0:
                assert native_result == raw_result, f"Mismatch for obj={obj}: native={native_result}, raw={raw_result}"
    finally:
        dist.destroy_process_group()


def test_gather_object_native_vs_raw() -> None:
    _run(_worker_gather_object_native_vs_raw)


# -- MultiPGUtil tests --


def _worker_multi_pg_util_all_reduce(rank: int, world_size: int) -> None:
    _init_gloo(rank, world_size)
    try:
        mesh = _make_mesh()
        inner_group = mesh.get_group("inner")
        outer_group = mesh.get_group("outer")

        # Step 1: single group
        tensor = torch.tensor([float(rank + 1)])
        MultiPGUtil.all_reduce(tensor, [inner_group], op=dist.ReduceOp.SUM)
        expected = {0: 3.0, 1: 3.0, 2: 7.0, 3: 7.0}[rank]
        assert tensor.item() == expected, f"rank {rank}: expected {expected}, got {tensor.item()}"

        # Step 2: two groups = global sum
        tensor = torch.tensor([float(rank + 1)])
        MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.SUM)
        assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

        # Step 3: bitwise equality across all ranks
        tensor = torch.tensor([float(rank + 1) * 0.1])
        MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.SUM)
        result_bytes = tensor.numpy().tobytes()
        gathered_bytes = [None] * world_size
        dist.all_gather_object(gathered_bytes, result_bytes)
        assert all(b == gathered_bytes[0] for b in gathered_bytes), "Not bitwise equal"

        # Step 4: empty groups = no-op
        tensor = torch.tensor([42.0])
        MultiPGUtil.all_reduce(tensor, [], op=dist.ReduceOp.SUM)
        assert tensor.item() == 42.0

        # Step 5: MAX op
        tensor = torch.tensor([float(rank + 1)])
        MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.MAX)
        assert tensor.item() == 4.0
    finally:
        dist.destroy_process_group()


def test_multi_pg_util_all_reduce() -> None:
    _run(_worker_multi_pg_util_all_reduce)


def _worker_multi_pg_util_gather_object(rank: int, world_size: int) -> None:
    _init_gloo(rank, world_size)
    try:
        mesh = _make_mesh()
        inner_group = mesh.get_group("inner")
        outer_group = mesh.get_group("outer")

        # Step 1: single group gather
        result = MultiPGUtil.gather_object({"rank": rank}, [inner_group])
        inner_rank = rank % 2
        if inner_rank == 0:
            assert result is not None
            assert len(result) == 2
            ranks_gathered = {item["rank"] for item in result}
            if rank == 0:
                assert ranks_gathered == {0, 1}
            else:
                assert ranks_gathered == {2, 3}
        else:
            assert result is None

        # Step 2: two group gather — global rank 0 gets everything
        result = MultiPGUtil.gather_object({"rank": rank}, [inner_group, outer_group])
        if rank == 0:
            assert result is not None
            assert len(result) == 4
            assert {item["rank"] for item in result} == {0, 1, 2, 3}
        else:
            assert result is None
    finally:
        dist.destroy_process_group()


def test_multi_pg_util_gather_object() -> None:
    _run(_worker_multi_pg_util_gather_object)


# -- _check_wait tests --


class TestCheckWait:
    def test_raises_on_false(self) -> None:
        work = MagicMock()
        work.wait.return_value = False

        with pytest.raises(RuntimeError, match="torchft allreduce failed"):
            _check_wait(work, "allreduce")

    def test_passes_on_true(self) -> None:
        work = MagicMock()
        work.wait.return_value = True

        _check_wait(work, "allreduce")

    def test_propagates_exception_from_wait(self) -> None:
        work = MagicMock()
        work.wait.side_effect = RuntimeError("NCCL timeout")

        with pytest.raises(RuntimeError, match="NCCL timeout"):
            _check_wait(work, "allreduce")
