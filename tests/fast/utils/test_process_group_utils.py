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


# -- Distributed test helpers --


class _DistTestBase:
    """Base class for tests that need a Gloo process group."""

    @staticmethod
    def _init_gloo(rank: int, world_size: int) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    @staticmethod
    def _cleanup() -> None:
        dist.destroy_process_group()

    @staticmethod
    def _run(fn, world_size: int = 4) -> None:
        mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)

    @staticmethod
    def _make_mesh(mesh_shape: tuple[int, ...] = (2, 2)):
        return init_device_mesh("cpu", mesh_shape=mesh_shape, mesh_dim_names=("outer", "inner"))


# -- GroupInfo tests (no distributed needed) --


class TestGroupInfo:
    def test_construction_with_none_group(self) -> None:
        info = GroupInfo(rank=0, size=4, group=None)
        assert info.rank == 0
        assert info.size == 4
        assert info.group is None

    def test_optional_fields_default_to_none(self) -> None:
        info = GroupInfo(rank=0, size=1, group=None)
        assert info.gloo_group is None
        assert info.src_rank is None


class TestGroupInfoVerification(_DistTestBase):
    @staticmethod
    def _worker_verify_passes(rank: int, world_size: int) -> None:
        _DistTestBase._init_gloo(rank, world_size)
        try:
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            GroupInfo(rank=rank, size=world_size, group=group)
        finally:
            _DistTestBase._cleanup()

    def test_verify_passes_with_correct_rank_size(self) -> None:
        self._run(self._worker_verify_passes)

    @staticmethod
    def _worker_verify_fails_rank(rank: int, world_size: int) -> None:
        _DistTestBase._init_gloo(rank, world_size)
        try:
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            wrong_rank = (rank + 1) % world_size
            try:
                GroupInfo(rank=wrong_rank, size=world_size, group=group)
                assert False, "Should have raised AssertionError"
            except AssertionError:
                pass
        finally:
            _DistTestBase._cleanup()

    def test_verify_fails_with_wrong_rank(self) -> None:
        self._run(self._worker_verify_fails_rank)


# -- GroupsInfo tests (no distributed needed) --


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
        assert result.groups_inner_to_outer == [None, None]

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


# -- GeneralPGUtil tests --


class TestGeneralPGUtil(_DistTestBase):
    @staticmethod
    def _worker_get_rank_and_size(rank: int, world_size: int) -> None:
        _DistTestBase._init_gloo(rank, world_size)
        try:
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            assert GeneralPGUtil.get_rank(group) == rank
            assert GeneralPGUtil.get_size(group) == world_size
        finally:
            _DistTestBase._cleanup()

    def test_get_rank_and_size(self) -> None:
        self._run(self._worker_get_rank_and_size)

    @staticmethod
    def _worker_all_reduce_sum(rank: int, world_size: int) -> None:
        _DistTestBase._init_gloo(rank, world_size)
        try:
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            tensor = torch.tensor([float(rank + 1)])
            GeneralPGUtil.all_reduce(tensor, group, op=dist.ReduceOp.SUM)
            assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0
        finally:
            _DistTestBase._cleanup()

    def test_all_reduce_sum(self) -> None:
        self._run(self._worker_all_reduce_sum)

    @staticmethod
    def _worker_reduce_to_root(rank: int, world_size: int) -> None:
        _DistTestBase._init_gloo(rank, world_size)
        try:
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            tensor = torch.tensor([float(rank + 1)])
            GeneralPGUtil.reduce(tensor, group, op=dist.ReduceOp.SUM)
            if rank == 0:
                assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0
        finally:
            _DistTestBase._cleanup()

    def test_reduce_to_root(self) -> None:
        self._run(self._worker_reduce_to_root)

    @staticmethod
    def _worker_broadcast_from_root(rank: int, world_size: int) -> None:
        _DistTestBase._init_gloo(rank, world_size)
        try:
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            tensor = torch.tensor([99.0]) if rank == 0 else torch.tensor([0.0])
            GeneralPGUtil.broadcast(tensor, group)
            assert tensor.item() == 99.0
        finally:
            _DistTestBase._cleanup()

    def test_broadcast_from_root(self) -> None:
        self._run(self._worker_broadcast_from_root)

    @staticmethod
    def _worker_is_native(rank: int, world_size: int) -> None:
        _DistTestBase._init_gloo(rank, world_size)
        try:
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            assert GeneralPGUtil.is_native(group)
        finally:
            _DistTestBase._cleanup()

    def test_is_native(self) -> None:
        self._run(self._worker_is_native)


# -- MultiPGUtil.all_reduce tests --


class TestMultiPGUtilAllReduce(_DistTestBase):
    """Test MultiPGUtil.all_reduce with 2D DeviceMesh (2x2).

    mesh = [[0, 1],    dim 0 ("outer") varies rows
            [2, 3]]    dim 1 ("inner") varies cols

    get_group("inner") (dim 1, same row):  {0, 1} and {2, 3}
    get_group("outer") (dim 0, same col):  {0, 2} and {1, 3}
    """

    @staticmethod
    def _worker_single_group(rank: int, world_size: int) -> None:
        """Single inner group == allreduce within row."""
        _DistTestBase._init_gloo(rank, world_size)
        try:
            mesh = _DistTestBase._make_mesh()
            inner_group = mesh.get_group("inner")

            tensor = torch.tensor([float(rank + 1)])
            MultiPGUtil.all_reduce(tensor, [inner_group], op=dist.ReduceOp.SUM)

            # inner groups (same row): {0,1} sum=1+2=3, {2,3} sum=3+4=7
            expected = {0: 3.0, 1: 3.0, 2: 7.0, 3: 7.0}[rank]
            assert tensor.item() == expected, f"rank {rank}: expected {expected}, got {tensor.item()}"
        finally:
            _DistTestBase._cleanup()

    def test_single_group(self) -> None:
        self._run(self._worker_single_group)

    @staticmethod
    def _worker_two_groups(rank: int, world_size: int) -> None:
        """[inner, outer] produces the global sum on all ranks."""
        _DistTestBase._init_gloo(rank, world_size)
        try:
            mesh = _DistTestBase._make_mesh()
            inner_group = mesh.get_group("inner")
            outer_group = mesh.get_group("outer")

            tensor = torch.tensor([float(rank + 1)])
            MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.SUM)

            assert tensor.item() == 1.0 + 2.0 + 3.0 + 4.0
        finally:
            _DistTestBase._cleanup()

    def test_two_groups(self) -> None:
        self._run(self._worker_two_groups)

    @staticmethod
    def _worker_bitwise_equal(rank: int, world_size: int) -> None:
        """All ranks get bitwise-identical results after multi-group allreduce."""
        _DistTestBase._init_gloo(rank, world_size)
        try:
            mesh = _DistTestBase._make_mesh()
            inner_group = mesh.get_group("inner")
            outer_group = mesh.get_group("outer")

            tensor = torch.tensor([float(rank + 1) * 0.1])
            MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.SUM)

            result_bytes = tensor.numpy().tobytes()
            gathered_bytes = [None] * world_size
            dist.all_gather_object(gathered_bytes, result_bytes)
            assert all(b == gathered_bytes[0] for b in gathered_bytes), "Results are not bitwise equal across ranks"
        finally:
            _DistTestBase._cleanup()

    def test_bitwise_equal(self) -> None:
        self._run(self._worker_bitwise_equal)

    @staticmethod
    def _worker_empty_groups(rank: int, world_size: int) -> None:
        """Empty groups list is a no-op."""
        _DistTestBase._init_gloo(rank, world_size)
        try:
            tensor = torch.tensor([42.0])
            MultiPGUtil.all_reduce(tensor, [], op=dist.ReduceOp.SUM)
            assert tensor.item() == 42.0
        finally:
            _DistTestBase._cleanup()

    def test_empty_groups(self) -> None:
        self._run(self._worker_empty_groups)

    @staticmethod
    def _worker_max_op(rank: int, world_size: int) -> None:
        """MAX op across two groups."""
        _DistTestBase._init_gloo(rank, world_size)
        try:
            mesh = _DistTestBase._make_mesh()
            inner_group = mesh.get_group("inner")
            outer_group = mesh.get_group("outer")

            tensor = torch.tensor([float(rank + 1)])
            MultiPGUtil.all_reduce(tensor, [inner_group, outer_group], op=dist.ReduceOp.MAX)

            assert tensor.item() == 4.0  # max of 1,2,3,4
        finally:
            _DistTestBase._cleanup()

    def test_max_op(self) -> None:
        self._run(self._worker_max_op)


# -- MultiPGUtil.gather_object tests --


class TestMultiPGUtilGatherObject(_DistTestBase):
    """Test MultiPGUtil.gather_object with 2D DeviceMesh."""

    @staticmethod
    def _worker_single_group(rank: int, world_size: int) -> None:
        """Gather within one group."""
        _DistTestBase._init_gloo(rank, world_size)
        try:
            mesh = _DistTestBase._make_mesh()
            inner_group = mesh.get_group("inner")

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
        finally:
            _DistTestBase._cleanup()

    def test_single_group(self) -> None:
        self._run(self._worker_single_group)

    @staticmethod
    def _worker_two_groups(rank: int, world_size: int) -> None:
        """Gather across inner then outer — global rank 0 gets all objects."""
        _DistTestBase._init_gloo(rank, world_size)
        try:
            mesh = _DistTestBase._make_mesh()
            inner_group = mesh.get_group("inner")
            outer_group = mesh.get_group("outer")

            result = MultiPGUtil.gather_object({"rank": rank}, [inner_group, outer_group])

            if rank == 0:
                assert result is not None
                assert len(result) == 4
                ranks_gathered = {item["rank"] for item in result}
                assert ranks_gathered == {0, 1, 2, 3}
            else:
                assert result is None
        finally:
            _DistTestBase._cleanup()

    def test_two_groups(self) -> None:
        self._run(self._worker_two_groups)
