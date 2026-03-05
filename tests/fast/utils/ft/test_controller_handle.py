"""Unit tests for ControllerHandleMixin.

Uses a minimal concrete subclass so the mixin is tested in isolation
rather than indirectly through each agent class.
"""

import time
from unittest.mock import MagicMock, patch

from miles.utils.ft.agents.controller_handle import ControllerHandleMixin


class _StubAgent(ControllerHandleMixin):
    def __init__(self) -> None:
        super().__init__()


class TestControllerHandleMixin:
    def test_caches_result(self) -> None:
        agent = _StubAgent()
        mock_handle = MagicMock()
        agent._controller_handle = mock_handle

        with patch("ray.get_actor") as mock_get_actor:
            result = agent._get_controller_handle()

            assert result is mock_handle
            mock_get_actor.assert_not_called()

    def test_negative_cache_within_cooldown(self) -> None:
        agent = _StubAgent()
        agent._last_lookup_failure_time = time.monotonic()

        with patch("ray.get_actor") as mock_get_actor:
            result = agent._get_controller_handle()

            assert result is None
            mock_get_actor.assert_not_called()

    def test_retries_after_cooldown(self) -> None:
        agent = _StubAgent()
        agent._last_lookup_failure_time = time.monotonic() - 60.0

        mock_handle = MagicMock()
        with patch("ray.get_actor", return_value=mock_handle):
            result = agent._get_controller_handle()

            assert result is mock_handle

    def test_reset(self) -> None:
        agent = _StubAgent()
        agent._controller_handle = MagicMock()
        agent._last_lookup_failure_time = time.monotonic()

        agent._reset_controller_handle()

        assert agent._controller_handle is None
        assert agent._last_lookup_failure_time is None
