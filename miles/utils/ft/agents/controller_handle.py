from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ControllerHandleMixin:
    """Lazy-caching lookup of the ft_controller Ray actor handle.

    Mixed into agent classes that need to communicate with FtController.
    Provides _get_controller_handle() and _reset_controller_handle().
    """

    def __init__(self) -> None:
        self._controller_handle: Any | None = None
        self._controller_lookup_failed: bool = False

    def _get_controller_handle(self) -> Any | None:
        if self._controller_handle is not None:
            return self._controller_handle
        if self._controller_lookup_failed:
            return None

        try:
            import ray

            self._controller_handle = ray.get_actor("ft_controller")
        except Exception:
            self._controller_lookup_failed = True
            logger.warning("Failed to get ft_controller actor handle", exc_info=True)
            return None

        return self._controller_handle

    def _reset_controller_handle(self) -> None:
        self._controller_handle = None
        self._controller_lookup_failed = False
