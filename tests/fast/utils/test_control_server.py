from typing import Literal
from unittest.mock import MagicMock

import httpx
import pytest

from miles.ray.train.group import RayTrainGroup
from miles.utils.control_server import (
    ActorCellHandle,
    CellHandle,
    CellInfo,
    ControlServer,
)


# ────────────────────── Mock cell handle ──────────────────────


class _MockCellHandle(CellHandle):
    def __init__(
        self,
        *,
        cell_id: str,
        cell_type: Literal["actor", "rollout"] = "actor",
        state: Literal["running", "stopped", "pending"] = "running",
    ) -> None:
        self._cell_id = cell_id
        self._cell_type = cell_type
        self._state = state
        self.stop_calls: list[int] = []
        self.start_calls: int = 0

    @property
    def cell_id(self) -> str:
        return self._cell_id

    @property
    def cell_type(self) -> Literal["actor", "rollout"]:
        return self._cell_type

    def get_info(self) -> CellInfo:
        return CellInfo(
            cell_id=self._cell_id,
            cell_type=self._cell_type,
            state=self._state,
            node_ids=[],
        )

    def stop(self, timeout_seconds: int) -> None:
        self.stop_calls.append(timeout_seconds)
        self._state = "stopped"

    def start(self) -> None:
        self.start_calls += 1
        self._state = "running"


# ────────────────────── Helpers ──────────────────────


def _make_server(*handles: CellHandle) -> ControlServer:
    server = ControlServer(port=0)
    for h in handles:
        server.register(h)
    return server


def _client(server: ControlServer) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=server._app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ────────────────────── Tests ──────────────────────


class TestGetHealth:
    @pytest.mark.anyio
    async def test_health_returns_ok(self):
        server = _make_server()
        async with _client(server) as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestGetCells:
    @pytest.mark.anyio
    async def test_returns_registered_cells(self):
        h1 = _MockCellHandle(cell_id="actor-0", cell_type="actor", state="running")
        h2 = _MockCellHandle(cell_id="actor-1", cell_type="actor", state="stopped")
        server = _make_server(h1, h2)

        async with _client(server) as client:
            resp = await client.get("/cells")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["cell_id"] == "actor-0"
        assert data[0]["state"] == "running"
        assert data[1]["cell_id"] == "actor-1"
        assert data[1]["state"] == "stopped"


class TestStopCell:
    @pytest.mark.anyio
    async def test_stop_success(self):
        h0 = _MockCellHandle(cell_id="actor-0", state="running")
        h1 = _MockCellHandle(cell_id="actor-1", state="running")
        server = _make_server(h0, h1)

        async with _client(server) as client:
            resp = await client.post("/cells/actor-0/stop")

        assert resp.status_code == 200
        assert h0.stop_calls == [30]

    @pytest.mark.anyio
    async def test_stop_unknown_cell_returns_404(self):
        server = _make_server()
        async with _client(server) as client:
            resp = await client.post("/cells/nonexistent/stop")

        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_stop_last_running_actor_returns_409(self):
        """Stopping the only running actor cell must be rejected with 409."""
        h0 = _MockCellHandle(cell_id="actor-0", state="running")
        h1 = _MockCellHandle(cell_id="actor-1", state="stopped")
        server = _make_server(h0, h1)

        async with _client(server) as client:
            resp = await client.post("/cells/actor-0/stop")

        assert resp.status_code == 409
        assert h0.stop_calls == []

    @pytest.mark.anyio
    async def test_stop_already_stopped_cell_succeeds(self):
        """Stopping an already-stopped cell is idempotent (200)."""
        h0 = _MockCellHandle(cell_id="actor-0", state="stopped")
        h1 = _MockCellHandle(cell_id="actor-1", state="running")
        server = _make_server(h0, h1)

        async with _client(server) as client:
            resp = await client.post("/cells/actor-0/stop")

        assert resp.status_code == 200


class TestStartCell:
    @pytest.mark.anyio
    async def test_start_success(self):
        h0 = _MockCellHandle(cell_id="actor-0", state="stopped")
        server = _make_server(h0)

        async with _client(server) as client:
            resp = await client.post("/cells/actor-0/start")

        assert resp.status_code == 200
        assert h0.start_calls == 1

    @pytest.mark.anyio
    async def test_start_unknown_cell_returns_404(self):
        server = _make_server()
        async with _client(server) as client:
            resp = await client.post("/cells/nonexistent/start")

        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_start_already_running_succeeds(self):
        """Starting an already-running cell is idempotent (200)."""
        h0 = _MockCellHandle(cell_id="actor-0", state="running")
        server = _make_server(h0)

        async with _client(server) as client:
            resp = await client.post("/cells/actor-0/start")

        assert resp.status_code == 200
        assert h0.start_calls == 1


# ────────────────────── ActorCellHandle unit tests ──────────────────────


class _MockRayTrainCell:
    """Minimal stand-in for RayTrainCell used by ActorCellHandle tests."""

    def __init__(self, *, is_running: bool = True, is_pending: bool = False) -> None:
        self._is_running = is_running
        self._is_pending = is_pending

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_pending(self) -> bool:
        return self._is_pending

    @property
    def is_stopped(self) -> bool:
        return not self._is_running and not self._is_pending


def _make_group_with_mock_cells(cells: list[_MockRayTrainCell]) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group._indep_dp_quorum_id = 0
    group._alive_cell_ids = frozenset()
    return group


class TestActorCellHandle:
    def test_cell_id_and_type(self):
        group = _make_group_with_mock_cells([_MockRayTrainCell()])
        handle = ActorCellHandle(group=group, cell_index=0)

        assert handle.cell_id == "actor-0"
        assert handle.cell_type == "actor"

    def test_get_info_running(self):
        group = _make_group_with_mock_cells([_MockRayTrainCell(is_running=True)])
        handle = ActorCellHandle(group=group, cell_index=0)

        info = handle.get_info()
        assert info.state == "running"
        assert info.cell_id == "actor-0"
        assert info.node_ids == []

    def test_get_info_pending(self):
        group = _make_group_with_mock_cells([_MockRayTrainCell(is_running=False, is_pending=True)])
        handle = ActorCellHandle(group=group, cell_index=0)

        info = handle.get_info()
        assert info.state == "pending"

    def test_get_info_stopped(self):
        group = _make_group_with_mock_cells([_MockRayTrainCell(is_running=False)])
        handle = ActorCellHandle(group=group, cell_index=0)

        info = handle.get_info()
        assert info.state == "stopped"

    def test_stop_delegates_to_group(self):
        group = _make_group_with_mock_cells([_MockRayTrainCell()])
        group.stop = MagicMock()
        handle = ActorCellHandle(group=group, cell_index=2)

        handle.stop(timeout_seconds=10)

        group.stop.assert_called_once_with(2)

    def test_start_delegates_to_group(self):
        group = _make_group_with_mock_cells([_MockRayTrainCell()])
        group.start = MagicMock()
        handle = ActorCellHandle(group=group, cell_index=1)

        handle.start()

        group.start.assert_called_once_with(1)
