import asyncio
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
    """A group of RayTrainCells, each an independent megatron instance."""

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

        self._indep_dp_quorum_id = 0

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

    def _assert_all_running_or_pending(self) -> None:
        for cell in self._cells:
            assert cell.is_running or cell.is_pending, (
                f"Cell {cell.cell_id} is stopped, all cells must be running or pending"
            )

    def _assert_all_running(self) -> None:
        for cell in self._cells:
            assert cell.is_running, f"Cell {cell.cell_id} is not running (state={cell._state.type})"

    def _has_pending_cells(self) -> bool:
        return any(cell.is_pending for cell in self._cells)

    def _refs_all_cells(self, fn_name: str, *args, **kwargs) -> list[ray.ObjectRef]:
        self._assert_all_running()
        return [ref for cell in self._cells for ref in cell.refs(fn_name, *args, **kwargs)]

    def _refs_first_cell(self, fn_name: str, *args, **kwargs) -> list[ray.ObjectRef]:
        self._assert_all_running()
        return self._cells[0].refs(fn_name, *args, **kwargs)

    # --- public sync API (unchanged signatures for callers) ---

    def async_init(self, args, role: str, with_ref: bool = False) -> list[ray.ObjectRef]:
        assert args is self.args
        self._init_with_ref = with_ref
        return self._refs_all_cells("init", args, role, with_ref=with_ref, indep_dp_quorum_id=self._indep_dp_quorum_id)

    def async_train(self, rollout_id: int, rollout_data_ref) -> list[ray.ObjectRef]:
        self._assert_all_running_or_pending()
        if self._has_pending_cells():
            asyncio.get_event_loop().run_until_complete(self._materialize_pending_cells())
        return self._refs_all_cells("train", rollout_id, rollout_data_ref)

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        ray.get(self._refs_first_cell("save_model", rollout_id, force_sync=force_sync))

    def update_weights(self) -> None:
        ray.get(self._refs_first_cell("update_weights"))

    def onload(self) -> None:
        ray.get(self._refs_all_cells("wake_up"))

    def offload(self) -> None:
        ray.get(self._refs_all_cells("sleep"))

    def clear_memory(self) -> None:
        ray.get(self._refs_all_cells("clear_memory"))

    def connect(self, critic_group: "RayTrainGroup") -> None:
        self._assert_all_running()
        assert len(self._cells) == len(critic_group._cells), (
            f"Actor and critic must have the same number of cells: "
            f"actor has {len(self._cells)}, critic has {len(critic_group._cells)}"
        )
        ray.get(
            [
                ref
                for cell, critic_cell in zip(self._cells, critic_group._cells, strict=True)
                for ref in cell.refs_connect(critic_cell)
            ]
        )

    def set_rollout_manager(self, rollout_manager) -> None:
        ray.get(self._refs_all_cells("set_rollout_manager", rollout_manager))

    def stop(self, cell_id: int) -> None:
        self._cells[cell_id].stop()

    def start(self, cell_id: int) -> None:
        """Mark a stopped cell as pending. Actual startup happens in async_train()."""
        cell = self._cells[cell_id]
        cell.mark_pending()

    async def _materialize_pending_cells(self) -> None:
        """Materialize all pending cells: recreate actors, reconfigure PGs, transfer ckpt.

        Flow:
        1. Recreate actors for each pending cell
        2. All running cells reconfigure indep_dp PG (new quorum_id)
        3. In parallel per pending cell:
           - Src cell sends ckpt
           - Pending cell init (blocks on recv until send arrives)
        """
        pending_cells = [cell for cell in self._cells if cell.is_pending]
        assert pending_cells, "No pending cells to materialize"

        running_cells = [cell for cell in self._cells if cell.is_running]
        assert running_cells, "No running cells to serve as checkpoint source"

        for cell in pending_cells:
            cell.recreate_actors()

        self._indep_dp_quorum_id += 1
        qid = self._indep_dp_quorum_id

        logger.info(
            f"Materializing pending cells {[c.cell_id for c in pending_cells]}, "
            f"indep_dp_quorum_id={qid}"
        )

        # Step 2: All previously-running cells reconfigure indep_dp PG
        reconfigure_refs = []
        for cell in running_cells:
            reconfigure_refs.extend(cell.refs("reconfigure_indep_dp", qid))
        await asyncio.gather(*reconfigure_refs)

        # Step 3: For each pending cell, send ckpt from a running cell + init with recv
        transfer_refs: list[ray.ObjectRef] = []
        for cell in pending_cells:
            src_cell = running_cells[0]
            transfer_refs.extend(src_cell.refs("send_ckpt", cell.cell_id))
            transfer_refs.extend(cell.refs(
                "init",
                self.args,
                cell.role,
                with_ref=self._init_with_ref,
                recv_ckpt_src_rank=src_cell.cell_id,
                indep_dp_quorum_id=qid,
            ))
        await asyncio.gather(*transfer_refs)

        logger.info(f"All pending cells materialized successfully")


PGTuple = tuple[PlacementGroup, list[int], list[int]]


def _slice_pg(pg: PGTuple, start: int, end: int) -> PGTuple:
    placement_group, bundle_indices, gpu_ids = pg
    return placement_group, bundle_indices[start:end], gpu_ids[start:end]


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
