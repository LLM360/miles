import logging
from typing import TYPE_CHECKING

import ray
from ray.util.placement_group import PlacementGroup

from miles.ray.train.cell import RayTrainCell
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_nodes: int,
        num_gpus_per_node: int,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
    ) -> None:
        self.args = args

        total_gpus = num_nodes * num_gpus_per_node
        num_cells = (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1
        gpus_per_cell = total_gpus // num_cells
        assert total_gpus % num_cells == 0, f"total_gpus ({total_gpus}) must be divisible by num_cells ({num_cells})"

        if num_cells > 1:
            self._indep_dp_store, indep_dp_store_addr = _create_tcp_store()
            logger.info(f"Created TCPStore for independent DP at {indep_dp_store_addr}")
        else:
            self._indep_dp_store, indep_dp_store_addr = None, None

        self._cells: list[RayTrainCell] = []
        for cell_id in range(num_cells):
            cell_pg = _slice_pg(pg, start=cell_id * gpus_per_cell, end=(cell_id + 1) * gpus_per_cell)
            self._cells.append(
                RayTrainCell(
                    args=args,
                    gpus_per_cell=gpus_per_cell,
                    pg=cell_pg,
                    num_gpus_per_actor=num_gpus_per_actor,
                    role=role,
                    cell_id=cell_id,
                    num_cells=num_cells,
                    indep_dp_store_addr=indep_dp_store_addr,
                )
            )

    def _execute(self, fn_name, *args, **kwargs):
        return ray.get(self._async_execute(fn_name, *args, **kwargs))

    def _execute_first_cell(self, fn_name, *args, **kwargs):
        return ray.get(self._cells[0].async_execute(fn_name, *args, **kwargs))

    def _async_execute(self, fn_name, *args, **kwargs):
        return [future for cell in self._cells for future in cell.async_execute(fn_name, *args, **kwargs)]

    def async_init(self, args, role: str, with_ref: bool = False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        assert args is self.args
        return self._async_execute("init", args, role, with_ref=with_ref)

    def async_train(self, rollout_id: int, rollout_data_ref):
        """Do one rollout training"""
        return self._async_execute("train", rollout_id, rollout_data_ref)

    def save_model(self, rollout_id: int, force_sync: bool = False):
        """Save actor model. Only cell 0 saves to avoid file write conflicts."""
        self._execute_first_cell("save_model", rollout_id, force_sync=force_sync)

    def update_weights(self):
        """Broadcast weights to rollout engines. Only cell 0 pushes (all cells have identical weights)."""
        self._execute_first_cell("update_weights")

    def onload(self):
        self._execute("wake_up")

    def offload(self):
        self._execute("sleep")

    def clear_memory(self):
        self._execute("clear_memory")

    def connect(self, critic_group: "RayTrainGroup"):
        assert len(self._cells) == len(critic_group._cells), (
            f"Actor and critic must have the same number of cells: "
            f"actor has {len(self._cells)}, critic has {len(critic_group._cells)}"
        )
        ray.get(
            [
                future
                for cell, critic_cell in zip(self._cells, critic_group._cells, strict=True)
                for future in cell.async_connect(critic_cell)
            ]
        )

    def set_rollout_manager(self, rollout_manager):
        self._execute("set_rollout_manager", rollout_manager)


PGTuple = tuple[PlacementGroup, list[int], list[int]]


def _slice_pg(pg: PGTuple, start: int, end: int) -> PGTuple:
    placement_group, bundle_indices, gpu_ids = pg
    return (placement_group, bundle_indices[start:end], gpu_ids[start:end])


def _create_tcp_store() -> tuple["torch.distributed.TCPStore", str]:
    import torch.distributed

    store = torch.distributed.TCPStore(
        host_name="0.0.0.0",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )
    host = ray.util.get_node_ip_address()
    port = store.port
    return store, f"{host}:{port}"
