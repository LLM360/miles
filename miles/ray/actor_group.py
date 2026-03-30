import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.megatron_args_utils import compute_megatron_dp_size


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

        num_cells = compute_megatron_dp_size(args) if args.independent_dp else 1
        total_gpus = num_nodes * num_gpus_per_node
        gpus_per_cell = total_gpus // num_cells
        assert total_gpus % num_cells == 0, (
            f"total_gpus ({total_gpus}) must be divisible by num_cells ({num_cells})"
        )

        self._cells: list[RayTrainCell] = []
        for cell_id in range(num_cells):
            cell_pg = _slice_pg(pg, start=cell_id * gpus_per_cell, end=(cell_id + 1) * gpus_per_cell)
            self._cells.append(RayTrainCell(
                args=args,
                gpus_per_cell=gpus_per_cell,
                pg=cell_pg,
                num_gpus_per_actor=num_gpus_per_actor,
                role=role,
                cell_id=cell_id,
                num_cells=num_cells,
            ))

    def _execute(self, fn_name, *args, **kwargs):
        return ray.get(self._async_execute(fn_name, *args, **kwargs))

    def _async_execute(self, fn_name, *args, **kwargs):
        return [
            future
            for cell in self._cells
            for future in cell.async_execute(fn_name, *args, **kwargs)
        ]

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
        """Save actor model."""
        self._execute("save_model", rollout_id, force_sync=force_sync)

    def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        self._execute("update_weights")

    def onload(self):
        self._execute("wake_up")

    def offload(self):
        self._execute("sleep")

    def clear_memory(self):
        self._execute("clear_memory")

    def connect(self, critic_group: "RayTrainGroup"):
        ray.get([
            future
            for cell, critic_cell in zip(self._cells, critic_group._cells, strict=True)
            for future in cell.async_connect(critic_cell)
        ])

    def set_rollout_manager(self, rollout_manager):
        self._execute("set_rollout_manager", rollout_manager)


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
    ) -> None:
        self.args = args
        self.cell_id = cell_id
        self.num_cells = num_cells
        self.role = role

        self._allocate_gpus_for_actor(gpus_per_cell, pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, gpus_per_cell: int, pg, num_gpus_per_actor):
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

        # Create worker actors
        self._actor_handles = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handles.append(actor)

    def async_execute(self, fn_name, *args, **kwargs):
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
