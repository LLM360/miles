import pytest

from miles.backends.sglang_utils.length_validation import validate_context_capacity


def test_context_and_actual_pool_both_cover_the_generation_window():
    info = {"context_length": 131072, "max_total_num_tokens": 196608, "page_size": 256}
    validate_context_capacity(info, required_context=131072)
    with pytest.raises(ValueError, match="context_length"):
        validate_context_capacity({**info, "context_length": 40960}, required_context=131072)
    # A pool equal to the context window still clips requests because admission needs spare pages.
    with pytest.raises(ValueError, match="KV capacity"):
        validate_context_capacity({**info, "max_total_num_tokens": 131072}, required_context=131072)
    with pytest.raises(ValueError, match="KV capacity"):
        validate_context_capacity({"context_length": 131072}, required_context=131072)
