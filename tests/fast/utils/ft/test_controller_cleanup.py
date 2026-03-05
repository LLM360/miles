"""Tests for FtController.run() cleanup paths and _execute_decision defensive branches."""
from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.models import ActionType, Decision
from tests.fast.utils.ft.conftest import (
    FakeNotifier,
    FixedDecisionDetector,
    make_test_controller,
)


class TestRunCleanupNotifierAclose:
    """Verify controller.run() handles notifier.aclose() exceptions gracefully."""

    @pytest.mark.anyio
    async def test_notifier_aclose_exception_does_not_propagate(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        assert harness.notifier is not None

        async def failing_aclose() -> None:
            raise RuntimeError("webhook connection broken")

        harness.notifier.aclose = failing_aclose  # type: ignore[assignment]

        async def _shutdown_soon() -> None:
            await asyncio.sleep(0.03)
            await harness.controller.shutdown()

        task = asyncio.create_task(_shutdown_soon())
        await harness.controller.run()
        await task

        assert harness.controller._shutting_down

    @pytest.mark.anyio
    async def test_exporter_stop_called_on_shutdown(self) -> None:
        harness = make_test_controller(tick_interval=0.01)

        stop_called = False
        original_stop = harness.controller_exporter.stop

        def tracking_stop() -> None:
            nonlocal stop_called
            stop_called = True
            original_stop()

        harness.controller_exporter.stop = tracking_stop  # type: ignore[assignment]

        async def _shutdown_soon() -> None:
            await asyncio.sleep(0.03)
            await harness.controller.shutdown()

        task = asyncio.create_task(_shutdown_soon())
        await harness.controller.run()
        await task

        assert stop_called


class TestExecuteDecisionUnknownAction:
    """Verify _execute_decision raises ValueError for unknown action types."""

    @pytest.mark.anyio
    async def test_unknown_action_type_raises(self) -> None:
        harness = make_test_controller()

        bogus_decision = Decision(
            action="totally_unknown_action",  # type: ignore[arg-type]
            reason="should not happen",
        )

        with pytest.raises(ValueError, match="Unknown action type"):
            await harness.controller._execute_decision(bogus_decision)
