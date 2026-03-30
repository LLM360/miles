import logging
import os
from typing import Literal, Union

import ray
from pydantic import ConfigDict
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


class _StatePending(StrictBaseModel):
    type: Literal["pending"] = "pending"


class _StateRunning(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    type: Literal["running"] = "running"
    actor_handles: list[ray.actor.ActorHandle]


class _StateStopped(StrictBaseModel):
    type: Literal["stopped"] = "stopped"


_CellState = Union[_StatePending, _StateRunning, _StateStopped]


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

        actor_handles = self._create_actors()
        self._state: _CellState = _StateRunning(actor_handles=actor_handles)

    @property
    def is_running(self) -> bool:
        return isinstance(self._state, _StateRunning)

    def _get_actor_handles(self) -> list[ray.actor.ActorHandle]:
        assert isinstance(self._state, _StateRunning), f"Cell {self.cell_id} is not running (state={self._state.type})"
        return self._state.actor_handles

    def stop(self) -> None:
        handles = self._get_actor_handles()
        for actor in handles:
            ray.kill(actor)
        self._state = _StateStopped()
        logger.info(f"Killed all actors in cell {self.cell_id}")

    def recreate_actors(self) -> None:
        assert isinstance(
            self._state, _StateStopped
        ), f"Cannot recreate actors for cell {self.cell_id} (state={self._state.type})"
        actor_handles = self._create_actors()
        self._state = _StateRunning(actor_handles=actor_handles)
        logger.info(f"Recreated actors for cell {self.cell_id}")

    def refs(self, fn_name: str, *args, **kwargs) -> list[ray.ObjectRef]:
        handles = self._get_actor_handles()
        return [getattr(actor, fn_name).remote(*args, **kwargs) for actor in handles]

    def refs_connect(self, critic_cell: "RayTrainCell") -> list[ray.ObjectRef]:
        handles = self._get_actor_handles()
        critic_handles = critic_cell._get_actor_handles()
        return [
            actor.connect_actor_critic.remote(critic) for actor, critic in zip(handles, critic_handles, strict=False)
        ]

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
