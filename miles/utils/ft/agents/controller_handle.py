from __future__ import annotations

import logging
import os
import time
from typing import Any

from miles.utils.ft.models import ft_controller_actor_name

logger = logging.getLogger(__name__)

_RETRY_COOLDOWN_SECONDS: float = 30.0


class ControllerHandleMixin:
    """Lazy-caching lookup of the ft_controller Ray actor handle.

    Mixed into agent classes that need to communicate with FtController.
    Provides _get_controller_handle() and _reset_controller_handle().

    After a failed lookup, retries are suppressed for _RETRY_COOLDOWN_SECONDS
    to avoid spamming logs, then the next call will attempt lookup again.
    """

    def __init__(self, ft_id: str = "") -> None:
        self._ft_id = ft_id or os.environ.get("FT_ID", "")
        self._controller_handle: Any | None = None
        self._last_lookup_failure_time: float | None = None

    def _get_controller_handle(self) -> Any | None:
        if self._controller_handle is not None:
            return self._controller_handle

        if self._last_lookup_failure_time is not None:
            elapsed = time.monotonic() - self._last_lookup_failure_time
            if elapsed < _RETRY_COOLDOWN_SECONDS:
                return None

        try:
            import ray

            actor_name = ft_controller_actor_name(self._ft_id)
            self._controller_handle = ray.get_actor(actor_name)
            self._last_lookup_failure_time = None
        except Exception:
            self._last_lookup_failure_time = time.monotonic()
            logger.warning("Failed to get ft_controller actor handle", exc_info=True)
            return None

        return self._controller_handle

    def _reset_controller_handle(self) -> None:
        self._controller_handle = None
        self._last_lookup_failure_time = None
