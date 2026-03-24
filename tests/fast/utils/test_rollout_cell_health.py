import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from miles.utils.rollout_cell_health import CellEntry, RolloutCellHealth


def _make_engine(*, healthy: bool = True, delay: float = 0.0) -> MagicMock:
    engine = MagicMock()
    if healthy:
        async def _remote() -> bool:
            if delay:
                await asyncio.sleep(delay)
            return True
        engine.health_generate.remote = _remote
    else:
        async def _remote_fail() -> bool:
            raise RuntimeError("engine dead")
        engine.health_generate.remote = _remote_fail
    return engine


def _make_cell(cell_id: str, engines: list[MagicMock]) -> CellEntry:
    return CellEntry(cell_id=cell_id, get_engines=lambda: engines)


@pytest.fixture()
def mock_gauge() -> MagicMock:
    with patch("miles.utils.rollout_cell_health.set_prometheus_gauge") as mock:
        yield mock


class TestCheckOneCell:
    @pytest.mark.asyncio()
    async def test_healthy_cell_reports_alive(self, mock_gauge: MagicMock) -> None:
        """When the lead engine's health_generate succeeds, gauge is set to 1.0."""
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealth(
            cells=[cell], session_id="sess-1", run_name="run-1", check_interval=100.0
        )
        checker.start()

        await asyncio.sleep(0.05)
        await checker.shutdown()

        mock_gauge.assert_called_with(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "run_name", "cell_id"],
            label_values=["sess-1", "run-1", "cell-0"],
            value=1.0,
        )

    @pytest.mark.asyncio()
    async def test_unhealthy_engine_reports_dead(self, mock_gauge: MagicMock) -> None:
        """When health_generate raises, gauge is set to 0.0."""
        engine = _make_engine(healthy=False)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealth(
            cells=[cell], session_id="sess-1", run_name="run-1", check_interval=100.0
        )
        checker.start()

        await asyncio.sleep(0.05)
        await checker.shutdown()

        mock_gauge.assert_called_with(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "run_name", "cell_id"],
            label_values=["sess-1", "run-1", "cell-0"],
            value=0.0,
        )

    @pytest.mark.asyncio()
    async def test_none_engine_reports_dead(self, mock_gauge: MagicMock) -> None:
        """When the lead engine is None, gauge is set to 0.0."""
        cell = _make_cell("cell-0", [None])  # type: ignore[list-item]

        checker = RolloutCellHealth(
            cells=[cell], session_id="sess-1", run_name="run-1", check_interval=100.0
        )
        checker.start()

        await asyncio.sleep(0.05)
        await checker.shutdown()

        mock_gauge.assert_called_with(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "run_name", "cell_id"],
            label_values=["sess-1", "run-1", "cell-0"],
            value=0.0,
        )

    @pytest.mark.asyncio()
    async def test_empty_engines_list_reports_dead(self, mock_gauge: MagicMock) -> None:
        """When get_engines returns an empty list, gauge is set to 0.0."""
        cell = CellEntry(cell_id="cell-0", get_engines=lambda: [])

        checker = RolloutCellHealth(
            cells=[cell], session_id="sess-1", run_name="run-1", check_interval=100.0
        )
        checker.start()

        await asyncio.sleep(0.05)
        await checker.shutdown()

        mock_gauge.assert_called_with(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "run_name", "cell_id"],
            label_values=["sess-1", "run-1", "cell-0"],
            value=0.0,
        )

    @pytest.mark.asyncio()
    async def test_timeout_reports_dead(self, mock_gauge: MagicMock) -> None:
        """When health_generate exceeds the timeout, gauge is set to 0.0."""
        engine = _make_engine(healthy=True, delay=5.0)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealth(
            cells=[cell], session_id="sess-1", run_name="run-1",
            check_interval=100.0, timeout=0.01,
        )
        checker.start()

        await asyncio.sleep(0.1)
        await checker.shutdown()

        mock_gauge.assert_called_with(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "run_name", "cell_id"],
            label_values=["sess-1", "run-1", "cell-0"],
            value=0.0,
        )

    @pytest.mark.asyncio()
    async def test_multiple_cells_checked_independently(self, mock_gauge: MagicMock) -> None:
        """Cell A healthy and cell B dead produce independent gauge updates."""
        engine_a = _make_engine(healthy=True)
        engine_b = _make_engine(healthy=False)
        cell_a = _make_cell("cell-a", [engine_a])
        cell_b = _make_cell("cell-b", [engine_b])

        checker = RolloutCellHealth(
            cells=[cell_a, cell_b], session_id="sess-1", run_name="run-1",
            check_interval=100.0,
        )
        checker.start()

        await asyncio.sleep(0.05)
        await checker.shutdown()

        calls = mock_gauge.call_args_list
        alive_call = call(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "run_name", "cell_id"],
            label_values=["sess-1", "run-1", "cell-a"],
            value=1.0,
        )
        dead_call = call(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "run_name", "cell_id"],
            label_values=["sess-1", "run-1", "cell-b"],
            value=0.0,
        )
        assert alive_call in calls
        assert dead_call in calls


class TestPauseResume:
    @pytest.mark.asyncio()
    async def test_pause_skips_check(self, mock_gauge: MagicMock) -> None:
        """When paused, no health checks or metric updates occur."""
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealth(
            cells=[cell], session_id="sess-1", run_name="run-1", check_interval=0.01,
        )
        checker.pause()
        checker.start()

        mock_gauge.reset_mock()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        mock_gauge.assert_not_called()

    @pytest.mark.asyncio()
    async def test_resume_re_enables_check(self, mock_gauge: MagicMock) -> None:
        """After resume, health checks fire again."""
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealth(
            cells=[cell], session_id="sess-1", run_name="run-1", check_interval=0.01,
        )
        checker.pause()
        checker.start()
        await asyncio.sleep(0.03)

        mock_gauge.reset_mock()
        checker.resume()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        assert mock_gauge.call_count > 0


class TestLifecycle:
    @pytest.mark.asyncio()
    async def test_shutdown_cancels_task(self, mock_gauge: MagicMock) -> None:
        """Shutdown cancels the background task cleanly."""
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealth(
            cells=[cell], session_id="sess-1", run_name="run-1", check_interval=0.01,
        )
        checker.start()

        await checker.shutdown()
        assert checker._task is not None and (checker._task.cancelled() or checker._task.done())

    @pytest.mark.asyncio()
    async def test_check_interval_respected(self, mock_gauge: MagicMock) -> None:
        """With a longer interval, fewer checks happen in the same wall time."""
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker_fast = RolloutCellHealth(
            cells=[cell], session_id="sess-1", run_name="run-1", check_interval=0.01,
        )
        checker_fast.start()
        await asyncio.sleep(0.1)
        await checker_fast.shutdown()
        fast_count = mock_gauge.call_count

        mock_gauge.reset_mock()

        checker_slow = RolloutCellHealth(
            cells=[cell], session_id="sess-2", run_name="run-1", check_interval=0.05,
        )
        checker_slow.start()
        await asyncio.sleep(0.1)
        await checker_slow.shutdown()
        slow_count = mock_gauge.call_count

        assert fast_count > slow_count
