def strip_param_name_prefix(name: str):
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name


def is_value_head_param_name(name: str) -> bool:
    """Return True if ``name`` is the shared-backbone critic value head."""
    return "value_head" in strip_param_name_prefix(name).split(".")


def zero_non_value_head_grads(model) -> None:
    """Zero backbone/LM-head grads so critic-only warmup does not step π."""
    print("@dhawgupta: zero backbone grads (critic-only warmup)", flush=True)
    modules = model if isinstance(model, (list, tuple)) else [model]
    for module in modules:
        for name, param in module.named_parameters():
            if is_value_head_param_name(name):
                continue
            if getattr(param, "main_grad", None) is not None:
                param.main_grad.zero_()
            if param.grad is not None:
                param.grad.zero_()
