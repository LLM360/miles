"""Math-Verify correctness reward for DAPO Math 17K.

The dataset labels are integer strings. Model responses are free-form derivations
whose final answer is expected in a LaTeX ``\\boxed{...}`` expression. Math-Verify
parses both sides before comparing them, so mathematically equivalent formatting
receives the same binary reward. Scoring runs in isolated subprocesses because
some symbolic comparisons can otherwise block a complete rollout indefinitely.
"""

import asyncio
import logging
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from examples.dapo_math_17k.reward_process import score_entrypoint
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from sympy import Basic, Sum

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_GOLD_EXTRACTION_CONFIG = (ExprExtractionConfig(),)
_PRED_EXTRACTION_CONFIG = (
    LatexExtractionConfig(
        boxed_match_priority=0,
        normalization_config=NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=False,
            nits=False,
            boxed="all",
            equations=False,
        ),
    ),
    ExprExtractionConfig(),
)

# Math-Verify implements POSIX timeouts with SIGALRM, which only works in a main
# interpreter thread. Each score therefore runs in a spawned process whose main
# thread can use this timeout safely. The parent hard timeout is deliberately
# larger than all three potentially timed operations: gold parsing, prediction
# parsing, and verification.
_MATH_VERIFY_TIMEOUT = 5
_REWARD_PROCESS_TIMEOUT = 20.0
_REWARD_CONCURRENCY = 16
_MAX_FINITE_SUM_TERMS = 256
_PROCESS_CONTEXT = multiprocessing.get_context("spawn")
_REWARD_EXECUTOR = ThreadPoolExecutor(max_workers=_REWARD_CONCURRENCY, thread_name_prefix="dapo-reward")


def _has_oversized_finite_sum(extractions: list[Any]) -> bool:
    """Return whether an extracted expression contains an unsafe finite sum."""
    for extraction in extractions:
        if not isinstance(extraction, Basic):
            continue

        for symbolic_sum in extraction.atoms(Sum):
            for limit in symbolic_sum.limits:
                if len(limit) != 3:
                    continue
                _, lower, upper = limit
                if lower.is_integer is not True or upper.is_integer is not True:
                    continue
                if lower.is_number is not True or upper.is_number is not True:
                    continue
                if int(upper - lower) + 1 > _MAX_FINITE_SUM_TERMS:
                    return True

    return False


@lru_cache(maxsize=4096)
def _parse_gold(label: str):
    return parse(
        label,
        extraction_config=_GOLD_EXTRACTION_CONFIG,
        parsing_timeout=_MATH_VERIFY_TIMEOUT,
    )


def _score_response(label: Any, response: str | None) -> float:
    """Score one response; this function must remain safe to spawn and pickle."""
    if label is None or not response:
        return 0.0

    try:
        gold = _parse_gold(str(label))
        prediction = parse(
            response,
            extraction_config=_PRED_EXTRACTION_CONFIG,
            parsing_timeout=_MATH_VERIFY_TIMEOUT,
        )
        if _has_oversized_finite_sum(prediction):
            logger.debug("Skipping Math-Verify for an oversized finite symbolic sum")
            return 0.0
        return float(
            bool(
                gold
                and prediction
                and verify(
                    gold,
                    prediction,
                    timeout_seconds=_MATH_VERIFY_TIMEOUT,
                )
            )
        )
    except Exception:
        logger.debug(
            "Math-Verify could not score response against label %r",
            label,
            exc_info=True,
        )
        return 0.0


def _terminate_process(process: multiprocessing.Process) -> None:
    """Stop a scoring process without leaving a live child behind."""
    if process.pid is None:
        return
    if not process.is_alive():
        process.join()
        return

    process.terminate()
    process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join()


def _run_in_isolated_process(
    target: Callable[..., Any],
    target_args: tuple[Any, ...],
    timeout_seconds: float,
    error_value: Any = 0.0,
) -> Any:
    """Run a score in a spawned process with a recipe-specific failure value."""
    parent_connection, child_connection = _PROCESS_CONTEXT.Pipe(duplex=False)
    process = _PROCESS_CONTEXT.Process(
        target=score_entrypoint,
        args=(child_connection, target, target_args, error_value),
        daemon=True,
    )

    try:
        process.start()
        child_connection.close()
        if parent_connection.poll(timeout_seconds):
            try:
                return parent_connection.recv()
            except EOFError:
                logger.warning("DAPO reward subprocess exited without returning a score")
                return error_value

        logger.warning("DAPO reward subprocess exceeded %.1fs", timeout_seconds)
        return error_value
    except Exception:
        logger.warning("DAPO reward subprocess failed", exc_info=True)
        return error_value
    finally:
        child_connection.close()
        parent_connection.close()
        _terminate_process(process)


async def _score_sample(sample: Sample) -> float:
    """Score one sample off the rollout event loop with a killable timeout."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _REWARD_EXECUTOR,
        _run_in_isolated_process,
        _score_response,
        (sample.label, sample.response),
        _REWARD_PROCESS_TIMEOUT,
    )


async def reward_func(args: Any, samples: Sample | list[Sample], **kwargs: Any) -> float | list[float]:
    """Return binary mathematical correctness in the original sample order."""
    if isinstance(samples, list):
        return await asyncio.gather(*(_score_sample(sample) for sample in samples))
    return await _score_sample(samples)
