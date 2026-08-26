from miles.utils.data import filter_long_prompt
from miles.utils.types import Sample


def test_filter_long_prompt_keeps_conversation_samples_when_template_is_deferred():
    samples = [Sample(prompt=[{"role": "user", "content": "question"}])]

    result = filter_long_prompt(samples, tokenizer=None, processor=None, max_length=8191)

    assert result is samples


def test_filter_long_prompt_keeps_samples_when_limit_is_disabled():
    samples = [Sample(prompt="question")]

    result = filter_long_prompt(samples, tokenizer=None, processor=None, max_length=None)

    assert result is samples
