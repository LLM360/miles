import asyncio
import random
import string
import threading
from copy import deepcopy

import orjson
import pytest

from miles.rollout.generate_utils.sample_utils import _merge_sample_pair
from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample, encode_samples
from miles.rollout.session.samples.merge import truncate_samples_by_total_tokens
from miles.utils.metric_utils import has_repetition
from miles.utils.repetition import (
    REPETITION_METRIC_KEY,
    record_repetition,
    repetition_metrics,
    sample_repetition,
)
from miles.utils.types import Sample


class CharacterTokenizer:
    def __init__(self):
        self.calls = []
        self.thread_ids = []

    def decode(self, tokens):
        self.calls.append(list(tokens))
        self.thread_ids.append(threading.get_ident())
        return "".join(map(chr, tokens))


def token_only(text, *, prompt="prompt", **kwargs):
    return Sample(
        tokens=list(map(ord, prompt + text)),
        response="",
        response_length=len(text),
        metadata={"response_decoded": False, "task_id": "task"},
        **kwargs,
    )


_rng = random.Random(7)
NON_REPEATING = "".join(_rng.choices(string.ascii_letters + string.digits, k=16000))


@pytest.mark.parametrize(
    "text",
    ["", "short", "x" * 10000, "x" * 10001, "repeat " * 2000, NON_REPEATING, "漢🙂" * 6000],
    ids=["empty", "short", "threshold", "above_threshold", "repeated", "varied", "unicode"],
)
async def test_worker_matches_existing_detector_without_filling_text(text):
    sample = token_only(
        text,
        reward=0.75,
        status=Sample.Status.COMPLETED,
        loss_mask=[1] * len(text),
        rollout_log_probs=[-0.5] * len(text),
    )
    before = deepcopy(sample.to_dict())
    tokenizer = CharacterTokenizer()

    await record_repetition(sample, tokenizer)

    assert sample_repetition(sample) is has_repetition(text)
    assert sample.metadata[REPETITION_METRIC_KEY]["repetitive"] is has_repetition(text)
    # Only the bounded telemetry field changed; tokens, masks, reward, status,
    # text and existing metadata remain identical.
    after = sample.to_dict()
    after["metadata"] = dict(after["metadata"])
    after["metadata"].pop(REPETITION_METRIC_KEY)
    assert after == before
    assert len(orjson.dumps(sample.metadata[REPETITION_METRIC_KEY])) < 200
    if text:
        assert tokenizer.calls == [list(map(ord, text))]
        assert len(tokenizer.thread_ids) == 1
        assert tokenizer.thread_ids[0] != threading.get_ident()
    else:
        assert tokenizer.calls == []  # No [-0:] prompt decoding.


async def test_repetitive_prompt_does_not_enter_response_metric():
    sample = token_only("answer", prompt="repeat " * 3000)
    await record_repetition(sample, CharacterTokenizer())
    assert sample_repetition(sample) is False


@pytest.mark.parametrize("mutation", ["append", "shorten", "replace_same_length"])
async def test_changed_tokens_make_cached_result_unknown(mutation):
    sample = token_only("x" * 10001)
    await record_repetition(sample, CharacterTokenizer())
    assert sample_repetition(sample) is True
    if mutation == "append":
        sample.tokens.append(ord("y"))
        sample.response_length += 1
    elif mutation == "shorten":
        sample.tokens.pop()
        sample.response_length -= 1
    else:
        sample.tokens[-1] = ord("y")
    assert sample_repetition(sample) is None
    assert repetition_metrics([sample]) == {"repetition_coverage": 0.0}


async def test_retry_drops_old_result_and_records_new_output():
    sample = token_only("x" * 10001)
    await record_repetition(sample, CharacterTokenizer())
    sample.reset_for_retry()
    assert REPETITION_METRIC_KEY not in sample.metadata
    assert sample.metadata["task_id"] == "task"
    sample.tokens = list(map(ord, "short"))
    sample.response_length = 5
    await record_repetition(sample, CharacterTokenizer())
    assert sample_repetition(sample) is False


@pytest.mark.parametrize("mode", ["sample_strip", "truncate"])
async def test_truncation_invalidates_and_recomputation_uses_final_tokens(mode):
    sample = token_only("x" * 10001, prompt="p", loss_mask=[1] * 10001, rollout_log_probs=[-0.25] * 10001)
    tokenizer = CharacterTokenizer()
    await record_repetition(sample, tokenizer)
    if mode == "sample_strip":
        sample.strip_last_output_tokens(10000, tokenizer)
    else:
        assert truncate_samples_by_total_tokens([sample], 2, tokenizer) == [sample]
    assert REPETITION_METRIC_KEY not in sample.metadata
    assert sample.tokens == [ord("p"), ord("x")]
    assert sample.loss_mask == [1]
    assert sample.rollout_log_probs == [-0.25]
    await record_repetition(sample, tokenizer)
    assert sample_repetition(sample) is False


async def test_merge_drops_per_turn_telemetry_without_metadata_mismatch():
    a = token_only("x", prompt="p", status=Sample.Status.COMPLETED)
    b = token_only("z", prompt="pxy", status=Sample.Status.COMPLETED)
    tokenizer = CharacterTokenizer()
    await record_repetition(a, tokenizer)
    await record_repetition(b, tokenizer)
    merged = _merge_sample_pair(a, b, tokenizer)
    assert REPETITION_METRIC_KEY not in merged.metadata
    assert REPETITION_METRIC_KEY in a.metadata  # Inputs weren't mutated.
    assert REPETITION_METRIC_KEY in b.metadata
    assert merged.tokens == list(map(ord, "pxyz"))
    assert merged.response == ""
    assert sample_repetition(merged) is None
    await record_repetition(merged, tokenizer)
    assert sample_repetition(merged) is False


async def test_result_survives_json_roundtrip():
    sample = token_only("x" * 10001)
    await record_repetition(sample, CharacterTokenizer())
    restored = Sample.from_dict(orjson.loads(orjson.dumps(sample.to_dict())))
    assert restored.response == ""
    assert sample_repetition(restored) is True


async def test_server_wire_preserves_valid_result_and_drops_stale_input_result():
    sample = token_only("x" * 10001)
    await record_repetition(sample, CharacterTokenizer())
    stale_input = Sample(metadata={REPETITION_METRIC_KEY: {"stale": True}})
    (restored,) = decode_samples_and_merge_input_sample(encode_samples([sample], {}), stale_input).samples
    assert restored.response == ""
    assert sample_repetition(restored) is True
    sample.metadata.pop(REPETITION_METRIC_KEY)
    (restored,) = decode_samples_and_merge_input_sample(encode_samples([sample], {}), stale_input).samples
    assert REPETITION_METRIC_KEY not in restored.metadata
    assert sample_repetition(restored) is None
    assert stale_input.metadata == {REPETITION_METRIC_KEY: {"stale": True}}


async def test_decoding_error_is_unknown_and_does_not_abort_sample(caplog):
    sample = token_only("answer", reward=0.75, status=Sample.Status.COMPLETED)
    sample.metadata[REPETITION_METRIC_KEY] = {"stale": True}

    class BrokenTokenizer:
        def decode(self, _tokens):
            raise RuntimeError("sensitive answer text must not be logged")

    await record_repetition(sample, BrokenTokenizer())
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 0.75
    assert sample_repetition(sample) is None
    assert REPETITION_METRIC_KEY not in sample.metadata
    assert "sensitive answer" not in caplog.text
    assert "Repetition telemetry unavailable (RuntimeError)" in caplog.text


async def test_cancellation_does_not_let_background_thread_write_result():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    class WaitingTokenizer:
        def decode(self, _tokens):
            started.set()
            try:
                assert release.wait(2)
                return "short"
            finally:
                finished.set()

    sample = token_only("answer")
    task = asyncio.create_task(record_repetition(sample, WaitingTokenizer()))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
    assert REPETITION_METRIC_KEY not in sample.metadata


@pytest.mark.parametrize("length", [-1, 100, True])
async def test_invalid_token_span_is_unknown(length):
    sample = token_only("answer")
    sample.response_length = length
    tokenizer = CharacterTokenizer()
    await record_repetition(sample, tokenizer)
    assert tokenizer.calls == []
    assert sample_repetition(sample) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 99),
        ("version", True),
        ("repetitive", 1),
        ("response_digest", "wrong"),
        ("response_length", True),
    ],
)
async def test_malformed_or_unknown_version_is_not_counted(field, value):
    sample = token_only("answer")
    await record_repetition(sample, CharacterTokenizer())
    sample.metadata[REPETITION_METRIC_KEY][field] = value
    assert sample_repetition(sample) is None


@pytest.mark.parametrize("payload", [None, True, "bad", [], {}])
def test_missing_or_invalid_result_is_unknown(payload):
    sample = token_only("answer")
    sample.metadata[REPETITION_METRIC_KEY] = payload
    assert sample_repetition(sample) is None


def test_legacy_readable_text_and_empty_answer_are_supported():
    assert sample_repetition(Sample(response="x" * 10001)) is True
    assert sample_repetition(Sample(response="short")) is False
    assert sample_repetition(Sample(response="")) is False
    assert sample_repetition(Sample(response=None)) is None


async def test_mixed_batch_denominator_uses_only_known_results():
    repeated = token_only("x" * 10001)
    await record_repetition(repeated, CharacterTokenizer())
    samples = [repeated, Sample(response="short"), token_only("unknown")]
    assert repetition_metrics(samples) == {"repetition_frac": 0.5, "repetition_coverage": 2 / 3}
    assert repetition_metrics(samples[2:]) == {"repetition_coverage": 0.0}
    assert repetition_metrics([]) == {"repetition_coverage": 0.0}
