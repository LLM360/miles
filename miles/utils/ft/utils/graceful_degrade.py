"""Decorator for graceful degradation: catch Exception, log, return default."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar, overload

_T = TypeVar("_T")


class FaultInjectionError(RuntimeError):
    """Raised by fault injection hooks during testing. Not caught by graceful_degrade."""


_ARG_REPR_LIMIT = 200
_SKIP_PARAMS = frozenset({"self", "cls"})


def _format_call_args(
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        parts: list[str] = []
        for name, value in bound.arguments.items():
            if name in _SKIP_PARAMS:
                continue
            raw = repr(value)
            if len(raw) > _ARG_REPR_LIMIT:
                raw = raw[:_ARG_REPR_LIMIT] + "..."
            parts.append(f"{name}={raw}")
        if parts:
            return " | " + ", ".join(parts)
    except Exception:
        pass
    return ""


@overload
def graceful_degrade(
    *,
    default: _T,
    msg: str | None = ...,
    log_level: int = ...,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]: ...


@overload
def graceful_degrade(
    *,
    msg: str | None = ...,
    log_level: int = ...,
) -> Callable[[Callable[..., _T]], Callable[..., _T | None]]: ...


def graceful_degrade(
    *,
    default: Any = None,
    msg: str | None = None,
    log_level: int = logging.WARNING,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a function so that any ``Exception`` is logged and *default* is returned.

    Supports both sync and async callables.  The log message defaults to
    ``"{func.__qualname__} failed"`` and always includes ``exc_info=True``.
    Call arguments (excluding self/cls) are automatically appended to the
    log message for diagnostics.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        base_msg = msg if msg is not None else f"{func.__qualname__} failed"
        func_logger = logging.getLogger(func.__module__)
        sig = inspect.signature(func)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except FaultInjectionError:
                    raise
                except Exception:
                    full_msg = base_msg + _format_call_args(sig, args, kwargs)
                    func_logger.log(log_level, full_msg, exc_info=True)
                    return default

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except FaultInjectionError:
                raise
            except Exception:
                full_msg = base_msg + _format_call_args(sig, args, kwargs)
                func_logger.log(log_level, full_msg, exc_info=True)
                return default

        return sync_wrapper

    return decorator
