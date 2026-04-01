import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING

import ray
from ray.util.placement_group import PlacementGroup

from miles.ray.train.cell import RayTrainCell, allocate_gpus_for_actor
from miles.utils.indep_dp import IndepDPInfo
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


class RayTrainGroup:
    """
    A group of ray actors

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
                    role=role,
                    with_ref=with_ref,
                    cell_index=cell_index,
                    rollout_manager=rollout_manager,
                    actor_factory=lambda: allocate_gpus_for_actor(
                        args=args,
                        gpus_per_cell=gpus_per_cell,
                        pg=cell_pg,
                        num_gpus_per_actor=num_gpus_per_actor,
                        indep_dp_store_addr=indep_dp_store_addr,
                    ),
                )
            )

        self._hb_stop_event: threading.Event | None = None
        self._hb_pause_event: threading.Event | None = None
        self._hb_thread: threading.Thread | None = None

        if len(self._cells) > 1:
            self._start_heartbeat_monitor()

    # ------------------------ APIs ------------------------

    async def init(self):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        refs = [
            future
            for cell in self._cells
            for future in cell.async_init(
                indep_dp_info=self._compute_indep_dp_info(
                    cell_index=cell.cell_index,
                    # all cells will be alive for this first initialization
                    alive_cell_indices=list(range(len(self._cells))),
                )
            )
        ]
        return await asyncio.gather(*refs)

    async def train(self, rollout_id: int, rollout_data_ref):
        """Do one rollout training"""
        await self._refresh_cells()
        await self._broadcast_alive("train", rollout_id, rollout_data_ref)

    async def save_model(self, rollout_id: int, force_sync: bool = False):
        """Save actor model. Only cell 0 saves to avoid file write conflicts."""
        await self._execute_first_alive("save_model", rollout_id, force_sync=force_sync)

    async def update_weights(self):
        """Broadcast weights to rollout engines."""
        # TODO: allow using all cells to update weights (instead of first alive cell)
        await self._execute_first_alive("update_weights")

    async def onload(self):
        await self._broadcast_alive("wake_up")
        self.resume_heartbeat()

    async def offload(self):
        self.pause_heartbeat()
        await self._broadcast_alive("sleep")

    async def clear_memory(self):
        await self._broadcast_alive("clear_memory")

    async def connect(self, critic_group: "RayTrainGroup"):
        assert len(self._cells) == len(critic_group._cells), (
            f"Actor and critic must have the same number of cells: "
            f"actor has {len(self._cells)}, critic has {len(critic_group._cells)}"
        )
        refs = [
            future
            for cell, critic_cell in zip(self._cells, critic_group._cells, strict=True)
            for future in cell.async_connect(critic_cell)
        ]
        await asyncio.gather(*refs)

    async def set_rollout_manager(self):
        refs = [future for cell in self._cells for future in cell.async_set_rollout_manager()]
        await asyncio.gather(*refs)

    def stop_cell(self, cell_index: int) -> None:
        self._cells[cell_index].stop()

    def start_cell(self, cell_index: int) -> None:
        """Mark a stopped cell as pending. Actual startup happens in train()."""
        self._cells[cell_index].mark_as_pending()

    def get_errored_cell_indices(self) -> list[int]:
        return [c.cell_index for c in self._cells if c.is_errored]

    # ------------------------ utils to forward calls to cells ------------------------

    def _dispatch_alive(self, fn_name: str, *args, **kwargs) -> list[tuple[RayTrainCell, list[ray.ObjectRef]]]:
        alive_cells = [c for c in self._cells if c.is_alive]
        assert alive_cells, "No alive cells"
        return [(cell, cell.async_execute(fn_name, *args, **kwargs)) for cell in alive_cells]

    async def _safe_await(self, futures_by_cell: list[tuple[RayTrainCell, list[ray.ObjectRef]]]) -> None:
        async def _await_cell(cell: RayTrainCell, futures: list[ray.ObjectRef]) -> None:
            try:
                await asyncio.gather(*futures)
            except Exception as e:
                logger.error(f"Cell {cell.cell_index} failed: {e}", exc_info=True)
                cell._mark_as_errored()

        await asyncio.gather(*[_await_cell(cell, fs) for cell, fs in futures_by_cell])

    async def _broadcast_alive(self, fn_name, *args, **kwargs):
        await self._safe_await(self._dispatch_alive(fn_name, *args, **kwargs))

    async def _execute_first_alive(self, fn_name, *args, **kwargs):
        alive_cells = [c for c in self._cells if c.is_alive]
        assert alive_cells, "No alive cells"
        for cell in alive_cells:
            try:
                await asyncio.gather(*cell.async_execute(fn_name, *args, **kwargs))
                return
            except Exception as e:
                logger.error(f"Cell {cell.cell_index} failed in {fn_name}: {e}", exc_info=True)
                cell._mark_as_errored()
        raise RuntimeError(f"All cells failed for {fn_name}")

    # ------------------------ internals for stop/start ------------------------

    async def _refresh_cells(self) -> None:
        snapshotted_pending_indices = [c.cell_index for c in self._cells if c.is_pending]
        snapshotted_alive_indices = [c.cell_index for c in self._cells if c.is_alive]
        will_alive_indices = sorted(list(set(snapshotted_pending_indices + snapshotted_alive_indices)))
        assert len(snapshotted_alive_indices) > 0, "Cannot recover when all cells are dead"

        # Step 0: Determine whether need to reconfigure
        exists_alive_cell_changed_config = any(
            cell.indep_dp_info.alive_cell_indices != will_alive_indices
            for cell in self._cells
            if cell.cell_index in snapshotted_alive_indices
        )
        exists_pending_cell = len(snapshotted_pending_indices) != 0
        needs_reconfigure = exists_pending_cell or exists_alive_cell_changed_config
        if not needs_reconfigure:
            return

        # Step 1: Bump states
        self._indep_dp_quorum_id += 1

        # Step 2: Allocate pending actors
        for c in self._cells:
            if c.cell_index in snapshotted_pending_indices:
                c.allocate_for_pending()

        # Step 3: Cooperatively prepare
        src_cell_index = snapshotted_alive_indices[0]  # TODO make it balanced, and support multi-src-to-one-dst
        src_alive_rank = will_alive_indices.index(src_cell_index)
        ckpt_dst_alive_ranks = [will_alive_indices.index(x) for x in snapshotted_pending_indices]

        try:
            await asyncio.gather(
                *[
                    (
                        c.prepare_indep_dp_mode_alive(
                            indep_dp_info=self._compute_indep_dp_info(
                                c.cell_index, alive_cell_indices=will_alive_indices
                            ),
                            send_ckpt_dst_ranks=ckpt_dst_alive_ranks if c.cell_index == src_cell_index else [],
                        )
                        if c.cell_index in snapshotted_alive_indices
                        else c.prepare_indep_dp_mode_healing(
                            indep_dp_info=self._compute_indep_dp_info(
                                c.cell_index, alive_cell_indices=will_alive_indices
                            ),
                            recv_ckpt_src_rank=src_alive_rank
                            if c.cell_index in snapshotted_pending_indices
                            else None,
                        )
                    )
                    for c in self._cells
                    if c.cell_index in will_alive_indices
                ]
            )
        except Exception:
            logger.exception("Failed to refresh cells, stopping pending cells")
            for c in self._cells:
                if c.cell_index in snapshotted_pending_indices:
                    c.stop()
            return

        assert [c.cell_index for c in self._cells if c.is_alive] == will_alive_indices

    def _compute_indep_dp_info(self, cell_index: int, alive_cell_indices: list[int]) -> IndepDPInfo:
        return IndepDPInfo(
            cell_index=cell_index,
            num_cells=len(self._cells),
            alive_rank=alive_cell_indices.index(cell_index),
            alive_size=len(alive_cell_indices),
            quorum_id=self._indep_dp_quorum_id,
            alive_cell_indices=alive_cell_indices,
        )

    # ------------------------ heartbeat monitor ------------------------

    def _start_heartbeat_monitor(self) -> None:
        self._hb_first_wait: float = self.args.trainer_heartbeat_first_wait
        self._hb_interval: float = self.args.trainer_heartbeat_interval
        self._hb_timeout: float = self.args.trainer_heartbeat_timeout
        self._hb_staleness: float = self.args.trainer_heartbeat_staleness

        self._hb_stop_event = threading.Event()
        self._hb_pause_event = threading.Event()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    def _heartbeat_loop(self) -> None:
        assert self._hb_stop_event is not None
        assert self._hb_pause_event is not None

        if self._hb_stop_event.wait(timeout=self._hb_first_wait):
            return
        while not self._hb_stop_event.is_set():
            if not self._hb_pause_event.is_set():
                self._check_heartbeats()
            if self._hb_stop_event.wait(timeout=self._hb_interval):
                break

    def _check_heartbeats(self) -> None:
        now = time.time()

        alive_cells = [cell for cell in self._cells if cell.is_alive]
        all_futures: list[tuple[RayTrainCell, ray.ObjectRef]] = []
        for cell in alive_cells:
            for actor in cell._get_actor_handles():
                all_futures.append((cell, actor.heartbeat.remote()))

        for cell, future in all_futures:
            if cell.is_errored:
                continue
            try:
                ts: float = ray.get(future, timeout=self._hb_timeout)
                if now - ts > self._hb_staleness:
                    logger.error(
                        f"Cell {cell.cell_index} heartbeat stale: "
                        f"last_active={ts:.1f}, now={now:.1f}, delta={now - ts:.1f}s"
                    )
                    cell._mark_as_errored()
            except Exception:
                logger.error(f"Cell {cell.cell_index} heartbeat failed", exc_info=True)
                cell._mark_as_errored()

    def stop_heartbeat(self) -> None:
        if self._hb_stop_event is not None:
            self._hb_stop_event.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=10.0)

    def pause_heartbeat(self) -> None:
        if self._hb_pause_event is not None:
            self._hb_pause_event.set()

    def resume_heartbeat(self) -> None:
        if self._hb_pause_event is not None:
            self._hb_pause_event.clear()


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
