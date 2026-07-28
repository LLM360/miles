import asyncio
import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


def merge_train_parallel_configs(configs: list[dict]) -> dict:
    if not configs:
        raise ValueError("at least one training-rank parallel config is required")

    size_keys = ("dp_size", "pp_size", "cp_size", "tp_size")
    merged = {key: configs[0][key] for key in size_keys}
    for config in configs[1:]:
        for key in size_keys:
            if config[key] != merged[key]:
                raise ValueError(
                    f"inconsistent {key}: rank 0 reported {merged[key]}, "
                    f"rank {config['world_rank']} reported {config[key]}"
                )

    routing_specs = {}
    routing_enabled = any(config["routing_replay_layer_indices"] is not None for config in configs)
    if routing_enabled:
        layers_by_pp_rank = {}
        for config in configs:
            layer_indices = config["routing_replay_layer_indices"]
            if layer_indices is None:
                raise ValueError(
                    f"rank {config['world_rank']} did not report routing-replay layers while other ranks did"
                )
            layer_indices = list(layer_indices)
            previous_layers = layers_by_pp_rank.setdefault(config["pp_rank"], layer_indices)
            if previous_layers != layer_indices:
                raise ValueError(
                    f"inconsistent routing-replay layers for PP rank {config['pp_rank']}: "
                    f"{previous_layers} != {layer_indices} on world rank {config['world_rank']}"
                )
            destination = (config["dp_rank"], config["pp_rank"], config["cp_rank"])
            spec = {
                "dp_rank": config["dp_rank"],
                "pp_rank": config["pp_rank"],
                "cp_rank": config["cp_rank"],
                "layer_indices": layer_indices,
            }
            previous = routing_specs.setdefault(destination, spec)
            if previous != spec:
                raise ValueError(f"inconsistent routing-replay shard spec for destination {destination}")

        expected_destinations = merged["dp_size"] * merged["pp_size"] * merged["cp_size"]
        if len(routing_specs) != expected_destinations:
            raise ValueError(
                f"expected {expected_destinations} routing-replay destinations, got {len(routing_specs)}"
            )

    merged["routing_replay_shard_specs"] = [
        routing_specs[key] for key in sorted(routing_specs)
    ]
    return merged


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
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        *,
        num_gpus_per_actor: float = 1,
        role: str,
        with_ref: bool,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self.with_ref = with_ref

        # Allocate the GPUs for actors w/o instantiating them
        self._actor_handles = self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

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
            from miles.backends.experimental.fsdp_utils import FSDPTrainRayActor

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
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            actor_handles.append(actor)

        return actor_handles

    async def init(self):
        """
        Allocate GPU resourced and initialize model, optimizer, local ckpt, etc.
        """
        return await self._broadcast("init", self.args, self.role, with_ref=self.with_ref)

    async def train(self, rollout_id, rollout_data_ref):
        """Do one rollout training"""
        await self.preload_rollout_data(rollout_id, rollout_data_ref)
        await self.train_preloaded(rollout_id)

    async def preload_rollout_data(self, rollout_id, rollout_data_ref):
        """Materialize rollout data on every rank without entering collectives."""
        refs = [
            actor.preload_rollout_data.remote(rollout_id, rollout_data_ref)
            for actor in self._actor_handles
        ]
        results = await asyncio.gather(*refs, return_exceptions=True)
        errors = [(rank, result) for rank, result in enumerate(results) if isinstance(result, BaseException)]
        if errors:
            # Avoid distributed cleanup after a partial preload failure.
            cleanup_refs = [
                actor.discard_preloaded_rollout.remote(rollout_id)
                for actor in self._actor_handles
            ]
            await asyncio.gather(*cleanup_refs, return_exceptions=True)
            details = "; ".join(f"rank {rank}: {error!r}" for rank, error in errors)
            raise RuntimeError(f"rollout {rollout_id} preload failed on {len(errors)} rank(s): {details}")
        return results

    async def train_preloaded(self, rollout_id):
        """Start training only after all ranks have acknowledged preload."""
        return await self._broadcast("train_preloaded", rollout_id)

    async def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        await self._broadcast("save_model", rollout_id, force_sync=force_sync)

    async def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        await self._broadcast("update_weights")

    async def onload(self):
        await self._broadcast("wake_up")

    async def offload(self):
        await self._broadcast("sleep")

    async def clear_memory(self):
        await self._broadcast("clear_memory")

    async def connect(self, critic_group):
        refs = [
            actor.connect_actor_critic.remote(critic)
            for actor, critic in zip(self._actor_handles, critic_group._actor_handles, strict=False)
        ]
        await asyncio.gather(*refs)

    async def set_rollout_manager(self, rollout_manager):
        await self._broadcast("set_rollout_manager", rollout_manager)
        configs = await self._broadcast("get_parallel_config")
        merged_config = merge_train_parallel_configs(configs)
        await rollout_manager.set_train_parallel_config.remote(merged_config)

    async def _broadcast(self, method_name: str, *args, **kwargs) -> list:
        refs = [getattr(actor, method_name).remote(*args, **kwargs) for actor in self._actor_handles]
        return await asyncio.gather(*refs)
