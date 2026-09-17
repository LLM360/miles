"""Validate a required rollout window against the engine's actual KV capacity."""


def validate_context_capacity(server_info: dict, *, required_context: int) -> None:
    if required_context <= 0:
        raise ValueError("rollout_required_context_len must be positive")
    context = server_info.get("context_length")
    capacity = server_info.get("max_total_num_tokens")
    page_size = server_info.get("page_size") or 1
    if context is None or context < required_context:
        raise ValueError(f"Rollout engine context_length={context} is below the required {required_context}")
    # Admission rounds the prompt up to a page and reserves a further page.
    minimum_capacity = required_context + 2 * page_size
    if capacity is None or capacity < minimum_capacity:
        raise ValueError(
            f"Rollout KV capacity={capacity} is below the required {minimum_capacity}; "
            "increase engine memory or tensor parallelism instead of silently shortening responses"
        )
