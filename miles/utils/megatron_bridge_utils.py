from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@contextmanager
def patch_megatron_model(model):
    unwrapped_models = unwrap_model(model)
    unwrapped_model = unwrapped_models[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    # Float16Module casts buffers to bf16, but expert_bias must stay fp32.
    # Restore before bridge export reads the values.
    for m in model:
        for module in m.modules():
            if hasattr(module, "_maintain_float32_expert_bias"):
                module._maintain_float32_expert_bias()

    # A shared critic is part of the native training checkpoint, but not the
    # HuggingFace causal LM. Bridge creates unmapped tasks for these parameters
    # and fails export. Hide only the critic during conversion; preserve the
    # exact modules/parameters (including optimizer references) on every exit.
    value_heads = []
    try:
        for module in unwrapped_models:
            if hasattr(module, "value_head"):
                value_heads.append((module, module.value_head))
                delattr(module, "value_head")
        yield
    finally:
        for module, head in value_heads:
            module.value_head = head
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
