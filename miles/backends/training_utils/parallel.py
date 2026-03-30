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


@dataclass(frozen=True)
class GroupsInfo:
    rank: int
    size: int
    groups: list[dist.ProcessGroup]


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
        groups: list[dist.ProcessGroup] = []
        if self.intra_dp_cp.size > 1:
            groups.append(self.intra_dp_cp.group)
        if self.indep_dp.size > 1:
            groups.append(self.indep_dp.group)
        return GroupsInfo(
            rank=self.indep_dp.rank * self.intra_dp_cp.size + self.intra_dp_cp.rank,
            size=self.indep_dp.size * self.intra_dp_cp.size,
            groups=groups,
        )


def all_reduce_multi(
    tensor: torch.Tensor,
    groups: Sequence[dist.ProcessGroup],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> None:
    for group in groups:
        dist.all_reduce(tensor, op=op, group=group)
