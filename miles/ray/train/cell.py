import logging
import os
from collections.abc import Callable

import ray
from pydantic import ConfigDict
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


class _StatePending(StrictBaseModel):
    pass


class _StateRunning(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    actor_handles: list[ray.actor.ActorHandle]


class _StateStopped(StrictBaseModel):
    pass


_CellState = _StatePending | _StateRunning | _StateStopped


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

        self._creation_kwargs = dict(
            gpus_per_cell=gpus_per_cell,
            pg=pg,
            num_gpus_per_actor=num_gpus_per_actor,
            indep_dp_store_addr=indep_dp_store_addr,
        )

        self._state: _CellState = _StatePending()
        self.allocate_for_pending()

    # ------------------------ lifecycle management ------------------------

    def stop(self) -> None:
        def _core():
            if self.is_running:
                handles = self._get_actor_handles()
                for actor in handles:
                    ray.kill(actor)

            return _StateStopped()

        self._change_state("stop", (_StatePending, _StateRunning), _core)

    def mark_as_pending(self) -> None:
        self._change_state("mark_as_pending", _StateStopped, _StatePending)

    def allocate_for_pending(self) -> None:
        def _core():
            actor_handles = self._allocate_gpus_for_actor(
                **self._creation_kwargs,
                args=self.args,
                cell_id=self.cell_id,
                num_cells=self.num_cells,
            )
            return _StateRunning(actor_handles=actor_handles)

        self._change_state("allocate_for_pending", _StatePending, _core)

    async def cooperatively_prepare(self):
        assert self.is_running

        if should_reconfigure_indep_dp:
            await reconfigure_indep_dp()

        if not is_initialized:
            await call_init()

        for dst_rank in send_ckpt_dst_ranks:
            await call_send_ckpt(dst_rank=dst_rank)

    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[_CellState] | tuple[type[_CellState], ...],
        fn: Callable[[], _CellState],
    ):
        logger.info(f"{debug_name} start {self.cell_id=}")
        assert isinstance(self._state, old_state_cls), f"{self.cell_id=} {self._state=}"
        self._state = fn()
        logger.info(f"{debug_name} end {self.cell_id=}")

    # ------------------------ actor creation ------------------------

    # TODO make it outside class
    @staticmethod
    def _allocate_gpus_for_actor(
        args,
        cell_id: int,
        num_cells: int,
        gpus_per_cell: int,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float,
        indep_dp_store_addr: str,
    ):
        world_size = gpus_per_cell

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **args.train_env_vars,
        }

        if source_patcher_config := args.dumper_source_patcher_config_train:
            env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

        if args.offload_train and args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        backend = args.train_backend
        if backend == "megatron":
            from miles.backends.megatron_utils.actor import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor

        else:
            from miles.backends.fsdp_utils import FSDPTrainRayActor

            actor_impl = FSDPTrainRayActor

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create worker actors
        actor_handles = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(
                world_size,
                rank,
                master_addr,
                master_port,
                cell_id=cell_id,
                num_cells=num_cells,
                indep_dp_store_addr=indep_dp_store_addr,
            )
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            actor_handles.append(actor)

        return actor_handles

    # ------------------------ usage ------------------------

    def async_execute(self, fn_name, *args, **kwargs):
        handles = self._get_actor_handles()
        return [getattr(actor, fn_name).remote(*args, **kwargs) for actor in handles]

    def async_connect(self, critic_cell: "RayTrainCell"):
        handles = self._get_actor_handles()
        critic_handles = critic_cell._get_actor_handles()
        return [
            actor.connect_actor_critic.remote(critic) for actor, critic in zip(handles, critic_handles, strict=False)
        ]

    # ------------------------ state helpers ------------------------

    @property
    def is_running(self) -> bool:
        return isinstance(self._state, _StateRunning)

    @property
    def is_pending(self) -> bool:
        return isinstance(self._state, _StatePending)

    def _get_actor_handles(self) -> list[ray.actor.ActorHandle]:
        assert isinstance(self._state, _StateRunning), f"Cell {self.cell_id} is not running (state={self._state.type})"
        return self._state.actor_handles
