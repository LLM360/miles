from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist


_parallel_state: "ParallelState | None" = None


def set_parallel_state(state: "ParallelState") -> None:
    global _parallel_state
    _parallel_state = state


def get_parallel_state() -> "ParallelState":
    assert _parallel_state is not None, "ParallelState not initialized. Call set_parallel_state() first."
    return _parallel_state


@dataclass
class ParallelState:
    """Core parallel state shared across all backends.
    Required by the general training utils.
    """

    intra_dp_rank: int
    intra_dp_cp_src_rank: int
    intra_dp_size: int
    cp_rank: int
    cp_size: int
    intra_dp_cp_rank: int
    intra_dp_cp_size: int
    intra_dp_group: dist.ProcessGroup | None
    intra_dp_cp_group: dist.ProcessGroup | None
    intra_dp_cp_group_gloo: dist.ProcessGroup | None
    cp_group: dist.ProcessGroup | None
    tp_size: int
    tp_rank: int
    tp_group: dist.ProcessGroup | None
    indep_dp_rank: int
    indep_dp_size: int
    indep_dp_group: "dist.ProcessGroup | None"
    is_pp_last_stage: bool = True
    vpp_size: int | None = 1
    microbatch_group_size_per_vp_stage: int | None = None

    def __post_init__(self) -> None:
        intra_trivial = self.intra_dp_rank == 0 and self.intra_dp_size == 1
        indep_trivial = self.indep_dp_rank == 0 and self.indep_dp_size == 1
        assert intra_trivial or indep_trivial, (
            f"intra_dp and indep_dp cannot both be non-trivial: "
            f"intra_dp=({self.intra_dp_rank}/{self.intra_dp_size}), "
            f"indep_dp=({self.indep_dp_rank}/{self.indep_dp_size})"
        )

    @property
    def effective_dp_rank(self) -> int:
        return self.indep_dp_rank if self.indep_dp_size > 1 else self.intra_dp_rank

    @property
    def effective_dp_size(self) -> int:
        return self.indep_dp_size if self.indep_dp_size > 1 else self.intra_dp_size

    @property
    def effective_dp_cp_rank(self) -> int:
        return self.indep_dp_rank * self.intra_dp_cp_size + self.intra_dp_cp_rank

    @property
    def effective_dp_cp_size(self) -> int:
        return self.indep_dp_size * self.intra_dp_cp_size

    @property
    def effective_dp_cp_group(self) -> list["dist.ProcessGroup"]:
        groups: list[dist.ProcessGroup] = []
        if self.intra_dp_cp_size > 1:
            groups.append(self.intra_dp_cp_group)
        if self.indep_dp_size > 1:
            groups.append(self.indep_dp_group)
        return groups

    @property
    def effective_dp_group(self) -> list["dist.ProcessGroup"]:
        groups: list[dist.ProcessGroup] = []
        if self.intra_dp_size > 1:
            groups.append(self.intra_dp_group)
        if self.indep_dp_size > 1:
            groups.append(self.indep_dp_group)
        return groups


def all_reduce_multi(
    tensor: torch.Tensor,
    groups: Sequence[dist.ProcessGroup],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> None:
    for group in groups:
        dist.all_reduce(tensor, op=op, group=group)
