"""Repetition telemetry for token-only rollouts, without transporting text.

The detector is unchanged: >10,000 characters and a zlib compression ratio
>10 on the final 10,000 characters. Decode once in the generation worker,
after final truncation. Readers validate the compact result against the
response tokens; missing/stale results are unknown, not negative detections.
"""

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

import orjson

from miles.utils.metric_utils import has_repetition

if TYPE_CHECKING:
    from miles.utils.types import Sample

logger = logging.getLogger(__name__)
REPETITION_METRIC_KEY = "repetition_metric"
REPETITION_METRIC_VERSION = 1


def invalidate_repetition(sample: "Sample") -> None:
    if isinstance(sample.metadata, dict):
        sample.metadata.pop(REPETITION_METRIC_KEY, None)


def _response_tokens(sample: "Sample") -> list[int]:
    n = sample.response_length
    if type(n) is not int or not 0 <= n <= len(sample.tokens):
        raise ValueError("invalid response_length for repetition telemetry")
    # [-0:] would include the prompt, which must never enter this metric.
    return sample.tokens[-n:] if n else []


def _fingerprint(tokens: list[int]) -> str:
    # Stable across processes/serialization, including same-length replacements.
    return hashlib.blake2b(orjson.dumps(tokens), digest_size=16).hexdigest()


def _measure(tokens: list[int], tokenizer) -> dict:
    text = tokenizer.decode(tokens) if tokens else ""
    return {
        "version": REPETITION_METRIC_VERSION,
        "response_length": len(tokens),
        "response_digest": _fingerprint(tokens),
        "repetitive": has_repetition(text),
    }


async def record_repetition(sample: "Sample", tokenizer) -> None:
    """Attach only telemetry. Never change response text or training fields.

    Work on a snapshot off the event loop; the background thread never mutates
    the sample. A decoding failure must not abort an otherwise usable rollout.
    """
    invalidate_repetition(sample)
    try:
        tokens = _response_tokens(sample)
        result = await asyncio.to_thread(_measure, tokens, tokenizer)
    except Exception as exc:
        logger.warning("Repetition telemetry unavailable (%s)", type(exc).__name__)
        return
    sample.metadata = {**(sample.metadata or {}), REPETITION_METRIC_KEY: result}


def sample_repetition(sample: "Sample") -> bool | None:
    """Read a validated worker result, or use readable text for legacy samples."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    decoded = metadata.get("response_decoded", True)
    if decoded is True:
        return has_repetition(sample.response) if isinstance(sample.response, str) else None
    if decoded is not False:
        return None

    result = metadata.get(REPETITION_METRIC_KEY)
    if not isinstance(result, dict):
        return None
    if type(result.get("version")) is not int or result["version"] != REPETITION_METRIC_VERSION:
        return None
    if type(result.get("repetitive")) is not bool:
        return None
    if type(result.get("response_length")) is not int or result["response_length"] != sample.response_length:
        return None
    try:
        if result.get("response_digest") != _fingerprint(_response_tokens(sample)):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return result["repetitive"]


def repetition_metrics(samples: list["Sample"], values: dict[int, bool | None] | None = None) -> dict[str, float]:
    """Fraction among known samples, with explicit coverage of the full group.

    Omit the fraction when every value is unknown; reporting zero would imply
    successful inspection. A supplied map avoids repeated hashing/detection
    across correctness/category splits in the same logging pass.
    """
    results = [sample_repetition(s) if values is None else values[id(s)] for s in samples]
    known = [result for result in results if result is not None]
    metrics = {"repetition_coverage": len(known) / len(samples) if samples else 0.0}
    if known:
        metrics["repetition_frac"] = sum(known) / len(known)
    return metrics
