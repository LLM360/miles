"""Tests for process_group_utils: GroupInfo, GroupsInfo, GeneralPGUtil, MultiPGUtil."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from miles.utils.process_group_utils import (
    GeneralPGUtil,
    GroupInfo,
    GroupsInfo,
    MultiPGUtil,
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
        assert info.src_rank is None


class TestGroupsInfo:
    def test_from_single(self) -> None:
        info = GroupInfo(rank=2, size=4, group=None)
        result = GroupsInfo.from_single(info)
        assert result.rank == 2
        assert result.size == 4
        assert result.groups_inner_to_outer == [None]

    def test_from_pair(self) -> None:
        inner = GroupInfo(rank=1, size=3, group=None)
        outer = GroupInfo(rank=2, size=4, group=None)
        result = GroupsInfo.from_pair(inner=inner, outer=outer)
        assert result.rank == 2 * 3 + 1  # 7
        assert result.size == 4 * 3  # 12

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


# -- Distributed tests (one spawn per class) --


def _worker_general_pg_util(rank: int, world_size: int) -> None:
    _init_gloo(rank, world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        # Step 1: get_rank / get_size / is_native
        assert GeneralPGUtil.get_rank(group) == rank
        assert GeneralPGUtil.get_size(group) == world_size
        assert GeneralPGUtil.is_native(group)

        # Step 2: GroupInfo verification passes with correct rank/size
        GroupInfo(rank=rank, size=world_size, group=group)

        # Step 3: GroupInfo verification fails with wrong rank
        wrong_rank = (rank + 1) % world_size
        try:
            GroupInfo(rank=wrong_rank, size=world_size, group=group)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

        # Step 4: all_reduce SUM
        tensor = torch.tensor([float(rank + 1)])
        GeneralPGUtil.all_reduce(tensor, group, op=dist.ReduceOp.SUM)
        assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

        # Step 5: reduce to root
        tensor = torch.tensor([float(rank + 1)])
        GeneralPGUtil.reduce(tensor, group, op=dist.ReduceOp.SUM)
        if rank == 0:
            assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0

        # Step 6: broadcast from root
        tensor = torch.tensor([99.0]) if rank == 0 else torch.tensor([0.0])
        GeneralPGUtil.broadcast(tensor, group)
        assert tensor.item() == 99.0
    finally:
        dist.destroy_process_group()


def test_general_pg_util() -> None:
    _run(_worker_general_pg_util)


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
