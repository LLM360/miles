def strip_param_name_prefix(name: str):
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name


def is_value_head_param_name(name: str) -> bool:
    """Return True if ``name`` is the shared-backbone critic value head."""
    return "value_head" in strip_param_name_prefix(name).split(".")
