import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


async def retry(
    fn: Callable[[], Awaitable[Any]],
) -> None:
    """Retry until ``fn`` does not throw."""
    attempt = 0
    while True:
        try:
            await fn()
            return
        except Exception:
            attempt += 1
            logger.warning(f"retry: attempt {attempt} failed, retrying", exc_info=True)
