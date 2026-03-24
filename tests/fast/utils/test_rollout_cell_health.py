from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.rollout_cell_health import CellEntry, RolloutCellHealthChecker


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


class _MockHandle:
    def __init__(self) -> None:
        self.set_gauge_calls: list[dict] = []

    class _RemoteProxy:
        def __init__(self, owner: _MockHandle) -> None:
            self._owner = owner

        def remote(self, name: str, value: float, extra_labels: dict[str, str] | None = None) -> str:
            self._owner.set_gauge_calls.append({"name": name, "value": value, "extra_labels": extra_labels})
            return "ref"

    @property
    def set_gauge(self) -> _RemoteProxy:
        return self._RemoteProxy(self)


@pytest.fixture()
def mock_prom() -> _MockHandle:
    handle = _MockHandle()
    with patch("miles.utils.rollout_cell_health.get_prometheus", return_value=handle):
        yield handle


def _find_call(handle: _MockHandle, cell_id: str) -> dict | None:
    for c in handle.set_gauge_calls:
        if c["extra_labels"] and c["extra_labels"].get("cell_id") == cell_id:
            return c
    return None


class TestCheckOneCell:
    @pytest.mark.asyncio()
    async def test_healthy_cell_reports_alive(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 1.0
        assert c["name"] == "miles_rollout_cell_alive"

    @pytest.mark.asyncio()
    async def test_unhealthy_engine_reports_dead(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=False)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_none_engine_reports_dead(self, mock_prom: _MockHandle) -> None:
        cell = _make_cell("cell-0", [None])  # type: ignore[list-item]

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_empty_engines_list_reports_dead(self, mock_prom: _MockHandle) -> None:
        cell = CellEntry(cell_id="cell-0", get_engines=lambda: [])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_timeout_reports_dead(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True, delay=5.0)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(
            cells=[cell], session_id="sess-1",
            check_interval=100.0, timeout=0.01,
        )
        checker.start()
        await asyncio.sleep(0.1)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_multiple_cells_checked_independently(self, mock_prom: _MockHandle) -> None:
        engine_a = _make_engine(healthy=True)
        engine_b = _make_engine(healthy=False)
        cell_a = _make_cell("cell-a", [engine_a])
        cell_b = _make_cell("cell-b", [engine_b])

        checker = RolloutCellHealthChecker(
            cells=[cell_a, cell_b], session_id="sess-1",
            check_interval=100.0,
        )
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        ca = _find_call(mock_prom, "cell-a")
        cb = _find_call(mock_prom, "cell-b")
        assert ca is not None and ca["value"] == 1.0
        assert cb is not None and cb["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_extra_labels_contain_session_id(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="my-sess", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["extra_labels"]["session_id"] == "my-sess"


class TestPauseResume:
    @pytest.mark.asyncio()
    async def test_pause_reports_minus_one(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell_a = _make_cell("cell-a", [engine])
        cell_b = _make_cell("cell-b", [engine])

        checker = RolloutCellHealthChecker(cells=[cell_a, cell_b], session_id="sess-1", check_interval=0.01)
        checker.pause()
        checker.start()

        await asyncio.sleep(0.05)
        await checker.shutdown()

        assert len(mock_prom.set_gauge_calls) > 0
        for call in mock_prom.set_gauge_calls:
            assert call["value"] == -1.0

        ca = _find_call(mock_prom, "cell-a")
        cb = _find_call(mock_prom, "cell-b")
        assert ca is not None
        assert cb is not None

    @pytest.mark.asyncio()
    async def test_resume_re_enables_check(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=0.01)
        checker.pause()
        checker.start()
        await asyncio.sleep(0.03)

        mock_prom.set_gauge_calls.clear()
        checker.resume()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        assert len(mock_prom.set_gauge_calls) > 0


class TestLifecycle:
    @pytest.mark.asyncio()
    async def test_shutdown_clears_task(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=0.01)
        checker.start()
        await checker.shutdown()

        assert checker._task is None

    @pytest.mark.asyncio()
    async def test_restart_after_shutdown(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await checker.shutdown()

        mock_prom.set_gauge_calls.clear()
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        assert len(mock_prom.set_gauge_calls) > 0

    @pytest.mark.asyncio()
    async def test_check_interval_respected(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker_fast = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=0.01)
        checker_fast.start()
        await asyncio.sleep(0.1)
        await checker_fast.shutdown()
        fast_count = len(mock_prom.set_gauge_calls)

        mock_prom.set_gauge_calls.clear()

        checker_slow = RolloutCellHealthChecker(cells=[cell], session_id="sess-2", check_interval=0.05)
        checker_slow.start()
        await asyncio.sleep(0.1)
        await checker_slow.shutdown()
        slow_count = len(mock_prom.set_gauge_calls)

        assert fast_count > slow_count
