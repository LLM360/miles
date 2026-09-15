# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import logging
from contextlib import nullcontext
from typing import Literal

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from miles.utils.misc import load_function
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.normal_(mean=0.0, std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


class SharedValueHead(torch.nn.Module):
    """Scalar critic on last-PP hidden states: linear or a small SiLU MLP."""

    def __init__(
        self,
        input_size: int,
        *,
        config: TransformerConfig,
        head_type: str = "linear",
        mlp_hidden_size: int | None = None,
        mlp_num_hidden_layers: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_parallel = config.sequence_parallel
        self.head_type = head_type
        hidden = mlp_hidden_size if mlp_hidden_size is not None else input_size
        layers: list[torch.nn.Module] = []
        if head_type == "linear":
            layers.append(torch.nn.Linear(input_size, 1, bias=bias))
        else:
            in_features = input_size
            for _ in range(mlp_num_hidden_layers):
                layers.append(torch.nn.Linear(in_features, hidden, bias=bias))
                layers.append(torch.nn.SiLU())
                in_features = hidden
            layers.append(torch.nn.Linear(in_features, 1, bias=bias))
        self.net = torch.nn.Sequential(*layers)
        if self.sequence_parallel:
            # Every Linear sees a sequence shard; Megatron all-reduces grads on
            # params marked sequence_parallel. Gather is after the net.
            for module in self.net.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.sequence_parallel = True
                    if module.bias is not None:
                        module.bias.sequence_parallel = True

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = self.net(input_).float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def attach_shared_value_head(model: GPTModel, args) -> GPTModel:
    """Add a scalar value head on the last pipeline stage.

    Default is ``g_phi(stopgrad(h_t))``. ``--share-backbone-critic-no-stopgrad``
    lets value grads reach the policy backbone. Policy logits still come from
    the attached hidden states.
    """
    if not getattr(model, "post_process", False):
        return model
    if getattr(model, "value_head", None) is not None:
        return model

    stopgrad = getattr(args, "share_backbone_critic_stopgrad", True)
    head_type = getattr(args, "share_backbone_critic_head_type", "linear")
    model.value_head = SharedValueHead(
        input_size=model.config.hidden_size,
        config=model.config,
        head_type=head_type,
        mlp_hidden_size=getattr(args, "share_backbone_critic_mlp_hidden_size", None),
        mlp_num_hidden_layers=getattr(args, "share_backbone_critic_mlp_num_hidden_layers", 1),
    )
    model.value_head.stopgrad_hidden = stopgrad
    original_postprocess = model._postprocess

    def _postprocess_with_value(hidden_states, *args, **kwargs):
        hidden = hidden_states.detach() if stopgrad else hidden_states
        values, _ = model.value_head(hidden)
        # Match GPTModel logits layout: [s, b, 1] -> [b, s, 1].
        model._last_values = values.transpose(0, 1).contiguous()
        return original_postprocess(hidden_states, *args, **kwargs)

    model._postprocess = _postprocess_with_value
    print("@dhawgupta: attach shared value_head", flush=True)
    return model


def maybe_attach_shared_value_head(model: GPTModel, args, role: str, post_process: bool) -> GPTModel:
    if post_process and role == "actor" and getattr(args, "share_backbone_critic", False):
        return attach_shared_value_head(model, args)
    return model


def _broadcast_replicated_param(tensor: torch.Tensor) -> None:
    if not torch.distributed.is_initialized():
        return
    if mpu.get_tensor_model_parallel_world_size() > 1:
        torch.distributed.broadcast(
            tensor, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group()
        )
    if mpu.get_data_parallel_world_size(with_context_parallel=True) > 1:
        torch.distributed.broadcast(
            tensor,
            src=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group(with_context_parallel=True),
        )


def maybe_reinit_zero_shared_value_head(model, force: bool = False) -> bool:
    """Re-init value_head after a policy-only load.

    Dist load ignores missing keys and can leave constructor weights in the
    module while optimizer/DDP main shards stay at zero. Do not skip just
    because the tensor is currently nonzero. ``force`` is for finetune /
    no-load-optim (no trained head in the ckpt). Without force, only re-init
    an all-zero head. Skip a resumed shared ckpt that already has a real head.
    Last PP stage only.
    """
    modules = model if isinstance(model, (list, tuple)) else [model]
    reinited = False
    for module in modules:
        inner = unwrap_to_inner_module(module)
        value_head = getattr(inner, "value_head", None)
        if value_head is None:
            continue
        if not force and any(p.detach().float().abs().max().item() > 0 for p in value_head.parameters()):
            continue
        print("@dhawgupta: reinit value_head after policy load", flush=True)
        value_head.reset_parameters()
        for param in value_head.parameters():
            _broadcast_replicated_param(param.data)
        reinited = True
    return reinited


def unwrap_to_inner_module(model: torch.nn.Module) -> torch.nn.Module:
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    return inner


def pop_last_values(model: torch.nn.Module) -> torch.Tensor | None:
    """Read and clear the value-head output stashed by the last forward."""
    inner = unwrap_to_inner_module(model)
    values = getattr(inner, "_last_values", None)
    if values is not None:
        inner._last_values = None
        print("@dhawgupta: pop_last_values", flush=True)
    return values


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            assert config is None, "miles builds the config from args, so it expects config to be None"
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            # Apply critic output layer if needed
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            return maybe_attach_shared_value_head(model, args, role, post_process)

        return wrapped_model_provider

    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # TODO: we should not manually set this...
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        provider.sequence_parallel = args.sequence_parallel
        provider.context_parallel_size = args.context_parallel_size
        provider.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        provider.variable_seq_lengths = args.variable_seq_lengths
        if hasattr(args, "moe_token_dispatcher_type"):
            provider.moe_token_dispatcher_type = args.moe_token_dispatcher_type
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        if getattr(args, "moe_router_bias_update_rate", None) is not None:
            provider.moe_router_bias_update_rate = args.moe_router_bias_update_rate
        if getattr(args, "moe_aux_loss_coeff", None) is not None:
            provider.moe_aux_loss_coeff = args.moe_aux_loss_coeff
        provider.finalize()

        def wrapped_bridge_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            assert config is None, "miles builds the config from args, so it expects config to be None"
            return provider.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        return wrapped_bridge_provider

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
        config: TransformerConfig | None = None,
        pg_collection=None,
    ) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        assert config is None, "miles builds the config from args, so it expects config to be None"
        config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    te_spec_kwargs = {
                        "num_experts": args.num_experts,
                        "moe_grouped_gemm": args.moe_grouped_gemm,
                        "qk_layernorm": args.qk_layernorm,
                        "multi_latent_attention": args.multi_latent_attention,
                        "moe_use_legacy_grouped_gemm": args.moe_use_legacy_grouped_gemm,
                    }
                    te_spec_params = inspect.signature(get_gpt_layer_with_transformer_engine_spec).parameters
                    if "fuse_layernorm_and_linear" in te_spec_params:
                        te_spec_kwargs["fuse_layernorm_and_linear"] = getattr(args, "layernorm_num_groups", 1) == 1
                    if "remap_unfused_layernorm_checkpoint_keys" in te_spec_params:
                        te_spec_kwargs["remap_unfused_layernorm_checkpoint_keys"] = (
                            getattr(args, "layernorm_num_groups", 1) == 1
                        )
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(**te_spec_kwargs)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            # hard code here to skip r3 registration for mtp layers
            # getattr is required to avoid ckpt conversion errors
            if getattr(args, "use_rollout_routing_replay", False):
                routing_replay_manager.enabled = False
                logger.warning(
                    "Rollout routing replay is not applicable for MTP modules, so skipped replay registration"
                )
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec
            if getattr(args, "use_rollout_routing_replay", False):
                routing_replay_manager.enabled = True

        with build_model_context(**build_model_context_args):
            model = GPTModel(**kwargs)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        return maybe_attach_shared_value_head(model, args, role, post_process)

    return model_provider
