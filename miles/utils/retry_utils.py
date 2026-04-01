import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def retry(
    fn: Callable[[], Awaitable[bool]],
) -> None:
    TODO
