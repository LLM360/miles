import logging
import os
from typing import TYPE_CHECKING

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
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

        self._quorum_id = 0

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

    def _assert_all_running(self) -> None:
        for cell in self._cells:
            assert cell.is_running, f"Cell {cell.cell_id} is stopped, all cells must be running"

    def _execute(self, fn_name, *args, **kwargs):
        return ray.get(self._async_execute(fn_name, *args, **kwargs))

    def _execute_first_cell(self, fn_name, *args, **kwargs):
        self._assert_all_running()
        return ray.get(self._cells[0].async_execute(fn_name, *args, **kwargs))

    def _async_execute(self, fn_name, *args, **kwargs):
        self._assert_all_running()
        return [future for cell in self._cells for future in cell.async_execute(fn_name, *args, **kwargs)]

    def async_init(self, args, role: str, with_ref: bool = False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        assert args is self.args
        return self._async_execute("init", args, role, with_ref=with_ref, quorum_id=self._quorum_id)

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
        self._assert_all_running()
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

    def stop(self, cell_id: int) -> None:
        """Stop a cell by killing all its actors."""
        cell = self._cells[cell_id]
        assert cell.is_running, f"Cell {cell_id} is already stopped"
        cell.stop()
        logger.info(f"Stopped cell {cell_id}")

    def start(self, cell_id: int, role: str, with_ref: bool = False) -> None:
        """Restart a stopped cell, recovering checkpoint from a healthy cell.

        The flow:
        1. Recreate actors for the stopped cell
        2. Increment quorum_id for fresh indep_dp PG
        3. In parallel:
           - New cell: init(recv_ckpt_src_rank=src_cell_id, quorum_id=new)
           - Src cell: reconfigure_indep_dp(quorum_id=new) then send_ckpt(dst)
           - Other cells: reconfigure_indep_dp(quorum_id=new)
        """
        target_cell = self._cells[cell_id]
        assert not target_cell.is_running, f"Cell {cell_id} is already running"

        src_cell_id = self._pick_healthy_cell(exclude=cell_id)
        target_cell.recreate_actors()

        self._quorum_id += 1
        logger.info(f"Starting cell {cell_id} from cell {src_cell_id}, quorum_id={self._quorum_id}")

        futures = []

        # Step 3a: New cell init (will recv ckpt from src_cell during init)
        futures.extend(
            target_cell.async_execute(
                "init",
                self.args,
                role,
                with_ref=with_ref,
                recv_ckpt_src_rank=src_cell_id,
                quorum_id=self._quorum_id,
            )
        )

        # Step 3b: Src cell reconfigure + send ckpt (must be parallel with 3a)
        src_cell = self._cells[src_cell_id]
        futures.extend(src_cell.async_execute("reconfigure_indep_dp_and_send_ckpt", self._quorum_id, cell_id))

        # Step 3c: Other healthy cells just reconfigure
        for cell in self._cells:
            if cell.cell_id != cell_id and cell.cell_id != src_cell_id and cell.is_running:
                futures.extend(cell.async_execute("reconfigure_indep_dp", self._quorum_id))

        ray.get(futures)
        logger.info(f"Cell {cell_id} started successfully")

    def _pick_healthy_cell(self, exclude: int) -> int:
        for cell in self._cells:
            if cell.cell_id != exclude and cell.is_running:
                return cell.cell_id
        raise RuntimeError(f"No healthy cell available (excluding cell {exclude})")


class RayTrainCell:
    def __init__(
        self,
        *,
        args,
        gpus_per_cell: int,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float,
        role: str,
        cell_id: int,
        num_cells: int,
        indep_dp_store_addr: str,
    ) -> None:
        self.args = args
        self.cell_id = cell_id
        self.num_cells = num_cells
        self.role = role

        self._gpus_per_cell = gpus_per_cell
        self._pg = pg
        self._num_gpus_per_actor = num_gpus_per_actor
        self._indep_dp_store_addr = indep_dp_store_addr

        self._actor_handles: list | None = self._create_actors()

    @property
    def is_running(self) -> bool:
        return self._actor_handles is not None

    def stop(self) -> None:
        assert self._actor_handles is not None
        for actor in self._actor_handles:
            ray.kill(actor)
        self._actor_handles = None
        logger.info(f"Killed all actors in cell {self.cell_id}")

    def recreate_actors(self) -> None:
        assert self._actor_handles is None, "Cannot recreate actors while cell is running"
        self._actor_handles = self._create_actors()
        logger.info(f"Recreated actors for cell {self.cell_id}")

    def _create_actors(self) -> list:
        world_size = self._gpus_per_cell

        assert self._pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = self._pg

        env_vars = {
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }

        if source_patcher_config := self.args.dumper_source_patcher_config_train:
            env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        backend = self.args.train_backend
        if backend == "megatron":
            from miles.backends.megatron_utils.actor import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor

        else:
            from miles.backends.fsdp_utils import FSDPTrainRayActor

            actor_impl = FSDPTrainRayActor

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        actor_handles = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=self._num_gpus_per_actor,
                num_gpus=self._num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(
                world_size,
                rank,
                master_addr,
                master_port,
                cell_id=self.cell_id,
                num_cells=self.num_cells,
                indep_dp_store_addr=self._indep_dp_store_addr,
            )
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            actor_handles.append(actor)

        return actor_handles

    def async_execute(self, fn_name, *args, **kwargs):
        assert self._actor_handles is not None, f"Cell {self.cell_id} is stopped"
        return [getattr(actor, fn_name).remote(*args, **kwargs) for actor in self._actor_handles]

    def async_connect(self, critic_group):
        return [
            actor.connect_actor_critic.remote(critic)
            for actor, critic in zip(self._actor_handles, critic_group._actor_handles, strict=False)
        ]


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
