from collections.abc import Callable

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm


from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, collect_named_tensors_for_weight_transfer


class BucketedWeightGatherMixin:
    """Mixin providing bucketed TP/EP all-gather and HF format conversion.

    Requires the consuming class to set:
        self.args: Namespace with update_weight_buffer_size (as the bucket size).
        self.model: Sequence[torch.nn.Module] (Megatron model chunks).
        self.model_name: str (for HF conversion).
        self.quantization_config: dict | None.
        self._is_source: bool (whether it's the rank broadcasting weights after `all_gather`).
    """

    def _gather_and_update_non_expert_weights(
        self,
        update_bucket_weight_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None],
        pbar: tqdm | None = None,
    ) -> None:
        """
        Bucketed TP all-gather + HF conversion for non-expert parameters.
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        After `all_gather`, update weights/buffer_size on source, do nothing on non-source.
        """

        buffer_size = 0
        converted_named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in collect_named_tensors_for_weight_transfer(self.args, self.model, is_expert=False):
            param = all_gather_param(self.args, name, param)
            if not self._is_source:
                continue

            param_size = param.numel() * param.element_size()
            if buffer_size + param_size > self.args.update_weight_buffer_size:
                update_bucket_weight_func(converted_named_tensors, pbar)
                converted_named_tensors = []
                buffer_size = 0

            converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            buffer_size += param_size

        if converted_named_tensors:
            update_bucket_weight_func(converted_named_tensors, pbar)

    def _gather_and_update_expert_weights(
        self,
        update_bucket_weight_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None],
        pbar: tqdm | None = None,
    ) -> None:
        """
        Bucketed TP + EP all-gather + HF conversion for expert parameters.
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in collect_named_tensors_for_weight_transfer(self.args, self.model, is_expert=True):
            param = all_gather_param(self.args, name, param)
            param_size = param.numel() * param.element_size()
            if (
                buffer_size + param_size
            ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size and named_tensors:
                self._update_expert_bucket_weights(named_tensors, update_bucket_weight_func, pbar)
                named_tensors = []
                buffer_size = 0

            named_tensors.append((name, param))
            buffer_size += param_size

        if named_tensors:
            self._update_expert_bucket_weights(named_tensors, update_bucket_weight_func, pbar)

    def _update_expert_bucket_weights(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        update_bucket_weight_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None],
        pbar: tqdm | None,
    ) -> None:
        """
        Gather EP → HF → update weights. Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names: list[list[str] | None] = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for ep_names in all_names:
            assert len(named_tensors) == len(
                ep_names
            ), f"mismatch names length: {len(named_tensors)} != {len(ep_names)}"

        all_gathered_params: list[list[tuple[str, torch.Tensor]]] = [
            [] for _ in range(mpu.get_expert_model_parallel_world_size())
        ]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, ep_names in enumerate(all_names):
                all_gathered_params[ep_rank].append((ep_names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_source:
            return

        flat_gathered = sum(all_gathered_params, [])

        converted_hf_tensors: list[tuple[str, torch.Tensor]] = []
        for name, param in flat_gathered:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        update_bucket_weight_func(converted_hf_tensors, pbar)
