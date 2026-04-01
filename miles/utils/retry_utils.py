import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def retry_until_true(
    fn: Callable[[], Awaitable[bool]],
    *,
    description: str = "retry_until_true",
) -> None:
    attempt = 0
    while not await fn():
        attempt += 1
        logger.warning(f"{description}: attempt {attempt} failed, retrying")
