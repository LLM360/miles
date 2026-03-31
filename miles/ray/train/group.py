import asyncio
import logging
from typing import TYPE_CHECKING

import ray
from ray.util.placement_group import PlacementGroup

from miles.ray.train.cell import RayTrainCell
from miles.utils.indep_dp import IndepDPInfo
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
        *,
        rollout_manager: object | None,
        num_gpus_per_actor: float = 1,
        role: str,
        with_ref: bool,
    ) -> None:
        self.args = args

        total_gpus = num_nodes * num_gpus_per_node
        num_cells = (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1
        gpus_per_cell = total_gpus // num_cells
        assert total_gpus % num_cells == 0, f"total_gpus ({total_gpus}) must be divisible by num_cells ({num_cells})"

        self._indep_dp_quorum_id = 0

        if num_cells > 1:
            self._indep_dp_store, indep_dp_store_addr = _create_tcp_store()
            logger.info(f"Created TCPStore for independent DP at {indep_dp_store_addr}")
        else:
            self._indep_dp_store, indep_dp_store_addr = None, None

        self._cells: list[RayTrainCell] = []
        for cell_index in range(num_cells):
            cell_pg = _slice_pg(pg, start=cell_index * gpus_per_cell, end=(cell_index + 1) * gpus_per_cell)
            self._cells.append(
                RayTrainCell(
                    args=args,
                    gpus_per_cell=gpus_per_cell,
                    pg=cell_pg,
                    num_gpus_per_actor=num_gpus_per_actor,
                    role=role,
                    with_ref=with_ref,
                    cell_index=cell_index,
                    indep_dp_store_addr=indep_dp_store_addr,
                    rollout_manager=rollout_manager,
                )
            )

    # ------------------------ APIs ------------------------

    def async_init(self):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        return [
            future
            for cell in self._cells
            for future in cell.async_init(indep_dp_info=self._compute_indep_dp_info(cell_index=cell.cell_index))
        ]

    def async_train(self, rollout_id: int, rollout_data_ref):
        """Do one rollout training"""
        asyncio.get_event_loop().run_until_complete(self._refresh_cells())
        return self._async_execute_alive("train", rollout_id, rollout_data_ref)

    def save_model(self, rollout_id: int, force_sync: bool = False):
        """Save actor model. Only cell 0 saves to avoid file write conflicts."""
        ray.get(self._async_execute_first_alive("save_model", rollout_id, force_sync=force_sync))

    def update_weights(self):
        """Broadcast weights to rollout engines. Only cell 0 pushes (all cells have identical weights)."""
        ray.get(self._async_execute_first_alive("update_weights"))

    def onload(self):
        ray.get(self._async_execute_alive("wake_up"))

    def offload(self):
        ray.get(self._async_execute_alive("sleep"))

    def clear_memory(self):
        ray.get(self._async_execute_alive("clear_memory"))

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

    def set_rollout_manager(self):
        ray.get([future for cell in self._cells for future in cell.async_set_rollout_manager()])

    def stop(self, cell_index: int) -> None:
        self._cells[cell_index].stop()

    def start(self, cell_index: int) -> None:
        """Mark a stopped cell as pending. Actual startup happens in async_train()."""
        self._cells[cell_index].mark_as_pending()

    # ------------------------ utils to forward calls to cells ------------------------

    def _async_execute_alive(self, fn_name, *args, **kwargs):
        return [
            future
            for cell in self._cells
            if cell.is_alive
            for future in cell.async_execute(fn_name, *args, **kwargs)
        ]

    def _async_execute_first_alive(self, fn_name, *args, **kwargs):
        running_cells = [cell for cell in self._cells if cell.is_alive]
        return running_cells[0].async_execute(fn_name, *args, **kwargs)

    # ------------------------ internals for stop/start ------------------------

    async def _refresh_cells(self) -> None:
        was_pending_indices = [cell.cell_index for cell in self._cells if cell.is_pending]
        was_alive_indices = [cell.cell_index for cell in self._cells if cell.is_alive]
        will_alive_indices = sorted(list(set(was_pending_indices + was_alive_indices)))
        assert len(was_alive_indices) > 0, "Cannot recover when all cells are dead"
        TODO_handle_zero_case

        # Step 1: Bump states
        self._indep_dp_quorum_id += 1

        # Step 2: Allocate actors
        for cell in self._cells:
            if cell.cell_index in was_pending_indices:
                cell.allocate_for_pending()

        # Step 3: Cooperatively prepare
        src_cell_index = was_alive_indices[0]  # TODO make it balanced, and support multi-src-to-one-dst
        await asyncio.gather(
            *[
                (
                    cell.prepare_indep_dp_mode_alive(
                        indep_dp_info=self._compute_indep_dp_info(cell.cell_index),
                        send_ckpt_dst_ranks=was_pending_indices if cell.cell_index == src_cell_index else [],
                    )
                    if cell.cell_index in was_alive_indices
                    else cell.prepare_indep_dp_mode_healing(
                        indep_dp_info=self._compute_indep_dp_info(cell.cell_index),
                        recv_ckpt_src_rank=src_cell_index if cell.cell_index in was_pending_indices else None,
                    )
                )
                for cell in self._cells
            ]
        )

    def _compute_indep_dp_info(self, cell_index: int) -> IndepDPInfo:
        return IndepDPInfo(
            cell_index=cell_index,
            num_cells=len(self._cells),
            alive_rank=alive_mapping[cell_index],
            alive_size=len(alive_mapping),
            quorum_id=self._indep_dp_quorum_id,
            alive_cell_indices=TODO,
        )


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
