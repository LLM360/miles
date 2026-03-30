from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _object_to_tensor, _tensor_to_object


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
        actual_rank = GeneralPGUtil.get_rank(group)
        actual_size = GeneralPGUtil.get_size(group)
        assert actual_rank == self.rank, f"{name}: rank mismatch: expected {self.rank}, got {actual_rank}"
        assert actual_size == self.size, f"{name}: size mismatch: expected {self.size}, got {actual_size}"


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


class GeneralPGUtil:
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

    @classmethod
    def gather_object(
        cls,
        obj: Any,
        object_gather_list: list[Any] | None,
        dst: int,
        group: dist.ProcessGroup,
    ) -> None:
        if cls.is_native(group):
            dist.gather_object(obj, object_gather_list, dst=dst, group=group)
        else:
            _gather_object_non_native(obj, object_gather_list, dst=dst, group=group)


class MultiPGUtil:
    """Operations across multiple process groups (inner-to-outer)."""

    @staticmethod
    def all_reduce(
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
            GeneralPGUtil.reduce(tensor, group, op)

        for group in reversed(groups_inner_to_outer):
            GeneralPGUtil.broadcast(tensor, group)

    @staticmethod
    def gather_object(
        obj: Any,
        groups_inner_to_outer: Sequence[dist.ProcessGroup],
    ) -> list[Any] | None:
        """Gather objects across multiple groups. Returns full list on rank 0, None on others."""
        objects = [obj]
        for group in groups_inner_to_outer:
            rank = GeneralPGUtil.get_rank(group)
            size = GeneralPGUtil.get_size(group)
            if rank == 0:
                gathered: list[Any] = [None] * size
                GeneralPGUtil.gather_object(objects, gathered, dst=0, group=group)
                objects = [item for sublist in gathered for item in sublist]
            else:
                GeneralPGUtil.gather_object(objects, None, dst=0, group=group)
                return None

        return objects


def _gather_object_non_native(
    obj: Any,
    object_gather_list: list[Any] | None,
    dst: int,
    group: dist.ProcessGroup,
) -> None:
    """gather_object for non-native (e.g. torchft) process groups.

    Copied from torch.distributed.distributed_c10d.gather_object (PyTorch v2.11.0)
    with the following modifications:
    - Replaced dist.get_rank()/get_world_size() with GeneralPGUtil (torchft PG
      returns wrong values from the C++ base class)
    - Replaced dist.all_gather()/dist.gather() with direct group.allgather()/
      group.gather() calls
    - Removed _rank_not_in_group check, _get_object_coll_device (use cpu),
      _validate_output_list_for_rank (inline assert), group_dst parameter
    - Simplified _object_to_tensor/_tensor_to_object calls (no group param
      needed since we always use cpu device)
    """
    # --- Begin: adapted from PyTorch v2.11.0 gather_object ---

    my_group_rank = GeneralPGUtil.get_rank(group)  # was: group.rank()
    if my_group_rank == dst:
        assert object_gather_list is not None
    else:
        assert object_gather_list is None

    current_device = torch.device("cpu")  # was: _get_object_coll_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = GeneralPGUtil.get_size(group)  # was: get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=torch.long, device=current_device)
    object_size_list = [object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    group.allgather([object_size_list], [[local_size]]).wait()  # was: all_gather(..., group=group)
    max_object_size = int(max(object_size_list).item())
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_group_rank == dst:
        coalesced_output_tensor = torch.empty(max_object_size * group_size, dtype=torch.uint8, device=current_device)
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)] for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    # was: gather(input_tensor, gather_list=..., group_dst=dst, group=group)
    if my_group_rank == dst:
        group.gather([output_tensors], [[input_tensor]], dist.GatherOptions(rootRank=dst)).wait()
    else:
        group.gather([], [[input_tensor]], dist.GatherOptions(rootRank=dst)).wait()

    if my_group_rank != dst:
        return

    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size)

    # --- End: adapted from PyTorch v2.11.0 gather_object ---
