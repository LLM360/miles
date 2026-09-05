from types import SimpleNamespace
from unittest.mock import Mock

from miles.utils.data import filter_long_prompt


def test_filter_long_prompt_preserves_samples_when_max_length_is_unset():
    samples = [SimpleNamespace(prompt="short"), SimpleNamespace(prompt="also short")]
    tokenizer = Mock()
    processor = Mock()

    result = filter_long_prompt(samples, tokenizer, processor, max_length=None)

    assert result is samples
    tokenizer.assert_not_called()
    processor.assert_not_called()


def test_filter_long_prompt_preserves_list_prompts_without_measuring_them():
    samples = [
        SimpleNamespace(prompt=[{"role": "user", "content": "hello"}]),
    ]
    tokenizer = Mock()
    processor = Mock()

    result = filter_long_prompt(samples, tokenizer, processor, max_length=128)

    assert result is samples
    tokenizer.assert_not_called()
    processor.assert_not_called()
