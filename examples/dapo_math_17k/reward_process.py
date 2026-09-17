"""Lightweight spawn entrypoint; reward children need no training/GPU imports."""

from collections.abc import Callable
from typing import Any


def score_entrypoint(
    connection: Any, target: Callable[..., Any], target_args: tuple[Any, ...], error_value: Any
) -> None:
    try:
        result = target(*target_args)
    except BaseException as error:
        result = error_value
        if isinstance(error_value, dict):
            result = {**error_value, "answer_reason": type(error).__name__}
    try:
        connection.send(result)
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()
