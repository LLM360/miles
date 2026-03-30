from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto

import torch
import torch.distributed as dist


_parallel_state: "ParallelState | None" = None


def set_parallel_state(state: "ParallelState") -> None:
    global _parallel_state
    _parallel_state = state


def get_parallel_state() -> "ParallelState":
    assert _parallel_state is not None, "ParallelState not initialized. Call set_parallel_state() first."
    return _parallel_state


class _DPMode(StrEnum):
    INTRA = auto()
    INDEP = auto()


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
        actual_rank = _GeneralProcessGroupUtil.get_rank(group)
        actual_size = _GeneralProcessGroupUtil.get_size(group)
        assert actual_rank == self.rank, f"{name}: rank mismatch: expected {self.rank}, got {actual_rank}"
        assert actual_size == self.size, f"{name}: size mismatch: expected {self.size}, got {actual_size}"

    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp) -> None:
        _GeneralProcessGroupUtil.all_reduce(tensor, self.group, op)


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



@dataclass
class ParallelState:
    """Core parallel state shared across all backends.
    Required by the general training utils.
    """

    intra_dp: GroupInfo
    intra_dp_cp: GroupInfo
    cp: GroupInfo
    tp: GroupInfo
    indep_dp: GroupInfo
    is_pp_last_stage: bool = True
    vpp_size: int | None = 1
    microbatch_group_size_per_vp_stage: int | None = None

    @property
    def _dp_mode(self):
        intra_trivial = self.intra_dp.rank == 0 and self.intra_dp.size == 1
        indep_trivial = self.indep_dp.rank == 0 and self.indep_dp.size == 1
        assert intra_trivial or indep_trivial, "intra_dp and indep_dp cannot both be non-trivial"

        return _DPMode.INTRA if indep_trivial else _DPMode.INDEP

    @property
    def effective_dp(self) -> GroupInfo:
        return {_DPMode.INTRA: self.intra_dp, _DPMode.INDEP: self.indep_dp}[self._dp_mode]

    @property
    def effective_dp_cp(self) -> GroupsInfo:
        return {
            _DPMode.INTRA: GroupsInfo.from_single(self.intra_dp_cp),
            _DPMode.INDEP: GroupsInfo.from_pair(inner=self.intra_dp_cp, outer=self.indep_dp),
        }[self._dp_mode]


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
        _GeneralProcessGroupUtil.reduce(tensor, group, op)

    for group in reversed(groups_inner_to_outer):
        _GeneralProcessGroupUtil.broadcast(tensor, group)


class _GeneralProcessGroupUtil:
    """Support both native ProcessGroup and torchft's custom process groups."""

    @classmethod
    def is_native(cls, group: dist.ProcessGroup) -> bool:
        return not hasattr(group, "_replica_id")

    @classmethod
    def get_rank(cls, group: dist.ProcessGroup) -> int:
        if cls.is_native(group):
            return dist.get_rank(group)
        else:
            return group._rank

    @classmethod
    def get_size(cls, group: dist.ProcessGroup) -> int:
        if cls.is_native(group):
            return dist.get_world_size(group)
        else:
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


