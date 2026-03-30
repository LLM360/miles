from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GroupInfo:
    rank: int
    size: int
    group: dist.ProcessGroup | None
    gloo_group: dist.ProcessGroup | None = None
    src_rank: int | None = None

    def __post_init__(self) -> None:
        self._verify_group(self.group, "group")
        self._verify_group(self.gloo_group, "gloo_group")

    def _verify_group(self, group: dist.ProcessGroup | None, name: str) -> None:
        if group is None:
            return
        actual_rank = GeneralProcessGroupUtil.get_rank(group)
        actual_size = GeneralProcessGroupUtil.get_size(group)
        assert actual_rank == self.rank, f"{name}: rank mismatch: expected {self.rank}, got {actual_rank}"
        assert actual_size == self.size, f"{name}: size mismatch: expected {self.size}, got {actual_size}"

    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp) -> None:
        GeneralProcessGroupUtil.all_reduce(tensor, self.group, op)


@dataclass(frozen=True)
class GroupsInfo:
    rank: int
    size: int
    groups_inner_to_outer: list[dist.ProcessGroup]

    @classmethod
    def from_single(cls, info: GroupInfo) -> "GroupsInfo":
        return cls(rank=info.rank, size=info.size, groups_inner_to_outer=[info.group])

    @classmethod
    def from_pair(cls, *, inner: GroupInfo, outer: GroupInfo) -> "GroupsInfo":
        return cls(
            rank=outer.rank * inner.size + inner.rank,
            size=outer.size * inner.size,
            groups_inner_to_outer=[inner.group, outer.group],
        )

    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp) -> None:
        _all_reduce_multi(tensor, self.groups_inner_to_outer, op)

    def gather_object(self, obj: Any, group_infos_inner_to_outer: list[GroupInfo]) -> list[Any] | None:
        """Gather objects across multiple groups. Returns full list on rank 0, None on others.

        Uses gloo_group from each GroupInfo for gather_object (which requires gloo).
        """
        assert len(group_infos_inner_to_outer) == len(self.groups_inner_to_outer)

        objects = [obj]
        for info in group_infos_inner_to_outer:
            assert info.gloo_group is not None, f"gloo_group required for gather_object, but {info} has None"
            if info.rank == 0:
                gathered: list[Any] = [None] * info.size
                dist.gather_object(objects, gathered, dst=0, group=info.gloo_group)
                objects = [item for sublist in gathered for item in sublist]
            else:
                dist.gather_object(objects, None, dst=0, group=info.gloo_group)
                return None

        return objects


class GeneralProcessGroupUtil:
    """Support both native ProcessGroup and torchft's custom process groups."""

    @classmethod
    def is_native(cls, group: dist.ProcessGroup) -> bool:
        return not hasattr(group, "_replica_id")

    @classmethod
    def get_rank(cls, group: dist.ProcessGroup) -> int:
        if cls.is_native(group):
            return dist.get_rank(group)
        return group._rank

    @classmethod
    def get_size(cls, group: dist.ProcessGroup) -> int:
        if cls.is_native(group):
            return dist.get_world_size(group)
        return group.size()

    @classmethod
    def all_reduce(cls, tensor: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp) -> None:
        if cls.is_native(group):
            dist.all_reduce(tensor, op=op, group=group)
        else:
            group.allreduce([tensor], dist.AllreduceOptions(reduceOp=op)).wait()

    @classmethod
    def reduce(cls, tensor: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp) -> None:
        if cls.is_native(group):
            dist.reduce(tensor, dst=0, op=op, group=group)
        else:
            group.reduce([tensor], dist.ReduceOptions(reduceOp=op, rootRank=0)).wait()

    @classmethod
    def broadcast(cls, tensor: torch.Tensor, group: dist.ProcessGroup) -> None:
        if cls.is_native(group):
            dist.broadcast(tensor, src=0, group=group)
        else:
            group.broadcast([tensor], dist.BroadcastOptions(rootRank=0)).wait()


def _all_reduce_multi(
    tensor: torch.Tensor,
    groups_inner_to_outer: Sequence[dist.ProcessGroup],
    op: dist.ReduceOp,
) -> None:
    """Reduce then broadcast across multiple groups for bitwise-equal results.

    Inner-to-outer reduce collapses values to the global root (rank 0 in every
    group). Outer-to-inner broadcast fans the result back out. Because broadcast
    is a pure copy, all ranks receive a bitwise-identical result regardless of
    floating-point non-determinism in the reduce path.
    """
    for group in groups_inner_to_outer:
        GeneralProcessGroupUtil.reduce(tensor, group, op)

    for group in reversed(groups_inner_to_outer):
        GeneralProcessGroupUtil.broadcast(tensor, group)
