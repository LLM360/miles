from __future__ import annotations

import asyncio

import httpx
import pytest

from miles.utils.control_server import (
    _create_control_app,
    _RolloutCellHandle,
    _CellRegistry,
    _ActorCellHandle,
)


class _MockHandle:
    def __init__(
        self,
        cell_id: str,
        cell_type: str,
        status: str = "running",
        node_ids: list[str] | None = None,
        stop_error: Exception | None = None,
        start_error: Exception | None = None,
    ) -> None:
        self.cell_id = cell_id
        self.cell_type = cell_type
        self._status = status
        self._node_ids = node_ids or []
        self._stop_error = stop_error
        self._start_error = start_error
        self.stop_calls: list[int] = []
        self.start_calls: int = 0

    async def stop(self, timeout_seconds: int) -> None:
        if self._stop_error:
            raise self._stop_error
        self.stop_calls.append(timeout_seconds)
        self._status = "stopped"

    async def start(self) -> None:
        if self._start_error:
            raise self._start_error
        self.start_calls += 1
        self._status = "running"

    async def get_status(self) -> str:
        return self._status

    async def get_node_ids(self) -> list[str]:
        return self._node_ids


class _MockRemoteCall:
    def __init__(self, return_value: object) -> None:
        self._return_value = return_value
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> asyncio.Future[object]:
        self.calls.append((args, kwargs))
        future: asyncio.Future[object] = asyncio.get_event_loop().create_future()
        future.set_result(self._return_value)
        return future


class _MockRolloutManager:
    def __init__(
        self,
        stop_return: object = None,
        start_return: object = None,
        status_return: str = "running",
    ) -> None:
        self.stop_cell = _MockRemoteCall(stop_return)
        self.start_cell = _MockRemoteCall(start_return)
        self.get_cell_status = _MockRemoteCall(status_return)


@pytest.fixture
def registry() -> _CellRegistry:
    return _CellRegistry()


@pytest.fixture
def async_client(registry: _CellRegistry) -> httpx.AsyncClient:
    app = _create_control_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


class TestCellRegistry:
    def test_register_and_get_by_id(self, registry: _CellRegistry) -> None:
        handle = _MockHandle(cell_id="cell-0", cell_type="rollout")
        registry.register(handle)
        assert registry.get("cell-0") is handle

    def test_get_unknown_id_raises_key_error(self, registry: _CellRegistry) -> None:
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_all_returns_all_registered(self, registry: _CellRegistry) -> None:
        h1 = _MockHandle(cell_id="cell-0", cell_type="rollout")
        h2 = _MockHandle(cell_id="cell-1", cell_type="rollout")
        registry.register(h1)
        registry.register(h2)

        all_handles = registry.get_all()
        assert len(all_handles) == 2
        assert h1 in all_handles
        assert h2 in all_handles

    def test_register_duplicate_id_raises(self, registry: _CellRegistry) -> None:
        h1 = _MockHandle(cell_id="cell-0", cell_type="rollout")
        h2 = _MockHandle(cell_id="cell-0", cell_type="rollout")
        registry.register(h1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(h2)


class TestGetCells:
    @pytest.mark.asyncio
    async def test_empty_registry(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.get("/cells")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_returns_all(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        registry.register(
            _MockHandle(
                cell_id="actor",
                cell_type="actor",
                status="running",
                node_ids=["node-0", "node-1"],
            )
        )
        registry.register(
            _MockHandle(
                cell_id="cell-0",
                cell_type="rollout",
                status="running",
                node_ids=["node-2"],
            )
        )
        registry.register(
            _MockHandle(
                cell_id="cell-1",
                cell_type="rollout",
                status="stopped",
                node_ids=["node-3"],
            )
        )

        resp = await async_client.get("/cells")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

        by_id = {d["cell_id"]: d for d in data}
        assert by_id["actor"]["cell_type"] == "actor"
        assert by_id["actor"]["status"] == "running"
        assert by_id["actor"]["node_ids"] == ["node-0", "node-1"]
        assert by_id["cell-0"]["cell_type"] == "rollout"
        assert by_id["cell-0"]["status"] == "running"
        assert by_id["cell-1"]["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_reflects_status_change(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = _MockHandle(cell_id="cell-0", cell_type="rollout", status="running")
        registry.register(handle)

        resp1 = await async_client.get("/cells")
        assert resp1.json()[0]["status"] == "running"

        handle._status = "stopped"

        resp2 = await async_client.get("/cells")
        assert resp2.json()[0]["status"] == "stopped"


class TestStopCell:
    @pytest.mark.asyncio
    async def test_default_timeout(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = _MockHandle(cell_id="cell-0", cell_type="rollout")
        registry.register(handle)

        resp = await async_client.post("/cells/cell-0/stop")
        assert resp.status_code == 200
        assert handle.stop_calls == [30]

    @pytest.mark.asyncio
    async def test_custom_timeout(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = _MockHandle(cell_id="cell-0", cell_type="rollout")
        registry.register(handle)

        resp = await async_client.post("/cells/cell-0/stop", json={"timeout_seconds": 60})
        assert resp.status_code == 200
        assert handle.stop_calls == [60]

    @pytest.mark.asyncio
    async def test_unknown_cell_returns_404(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.post("/cells/nonexistent/stop")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_already_stopped_is_idempotent(
        self, registry: _CellRegistry, async_client: httpx.AsyncClient
    ) -> None:
        handle = _MockHandle(cell_id="cell-0", cell_type="rollout", status="stopped")
        registry.register(handle)

        resp = await async_client.post("/cells/cell-0/stop")
        assert resp.status_code == 200
        assert len(handle.stop_calls) == 1

    @pytest.mark.asyncio
    async def test_handle_raises_returns_500(
        self, registry: _CellRegistry, async_client: httpx.AsyncClient
    ) -> None:
        handle = _MockHandle(
            cell_id="cell-0",
            cell_type="rollout",
            stop_error=RuntimeError("engine crashed"),
        )
        registry.register(handle)

        resp = await async_client.post("/cells/cell-0/stop")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_training_cell_returns_error(
        self, registry: _CellRegistry, async_client: httpx.AsyncClient
    ) -> None:
        registry.register(_ActorCellHandle(node_ids=["n0"]))

        resp = await async_client.post("/cells/training/stop")
        assert resp.status_code == 500
        assert "managed by the platform" in resp.json()["detail"]


class TestStartCell:
    @pytest.mark.asyncio
    async def test_calls_handle(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = _MockHandle(cell_id="cell-0", cell_type="rollout", status="stopped")
        registry.register(handle)

        resp = await async_client.post("/cells/cell-0/start")
        assert resp.status_code == 200
        assert handle.start_calls == 1

    @pytest.mark.asyncio
    async def test_unknown_cell_returns_404(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.post("/cells/nonexistent/start")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_already_running_is_idempotent(
        self, registry: _CellRegistry, async_client: httpx.AsyncClient
    ) -> None:
        handle = _MockHandle(cell_id="cell-0", cell_type="rollout", status="running")
        registry.register(handle)

        resp = await async_client.post("/cells/cell-0/start")
        assert resp.status_code == 200
        assert handle.start_calls == 1

    @pytest.mark.asyncio
    async def test_handle_raises_returns_500(
        self, registry: _CellRegistry, async_client: httpx.AsyncClient
    ) -> None:
        handle = _MockHandle(
            cell_id="cell-0",
            cell_type="rollout",
            start_error=RuntimeError("engine crashed"),
        )
        registry.register(handle)

        resp = await async_client.post("/cells/cell-0/start")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_training_cell_returns_error(
        self, registry: _CellRegistry, async_client: httpx.AsyncClient
    ) -> None:
        registry.register(_ActorCellHandle(node_ids=["n0"]))

        resp = await async_client.post("/cells/training/start")
        assert resp.status_code == 500
        assert "managed by the platform" in resp.json()["detail"]


class TestRolloutCellHandle:
    @pytest.mark.asyncio
    async def test_stop_delegates_to_manager(self) -> None:
        manager = _MockRolloutManager()

        handle = _RolloutCellHandle(rollout_manager=manager, cell_id="cell-0")
        await handle.stop(timeout_seconds=45)

        assert manager.stop_cell.calls == [(("cell-0", 45), {})]

    @pytest.mark.asyncio
    async def test_start_delegates_to_manager(self) -> None:
        manager = _MockRolloutManager()

        handle = _RolloutCellHandle(rollout_manager=manager, cell_id="cell-0")
        await handle.start()

    @pytest.mark.asyncio
    async def test_get_status_delegates_to_manager(self) -> None:
        manager = _MockRolloutManager(status_return="stopped")

        handle = _RolloutCellHandle(rollout_manager=manager, cell_id="cell-0")
        assert await handle.get_status() == "stopped"

    @pytest.mark.asyncio
    async def test_get_node_ids_delegates_to_manager(self) -> None:
        manager = _MockRolloutManager()
        manager.get_cell_node_ids = _MockRemoteCall(["n0", "n1"])

        handle = _RolloutCellHandle(rollout_manager=manager, cell_id="cell-0")
        assert await handle.get_node_ids() == ["n0", "n1"]

    def test_cell_type_is_rollout(self) -> None:
        handle = _RolloutCellHandle(rollout_manager=object(), cell_id="cell-0")
        assert handle.cell_type == "rollout"
        assert handle.cell_id == "cell-0"


class TestActorCellHandle:
    @pytest.mark.asyncio
    async def test_stop_raises(self) -> None:
        handle = _ActorCellHandle(node_ids=["n0"])
        with pytest.raises(NotImplementedError, match="managed by the platform"):
            await handle.stop(timeout_seconds=30)

    @pytest.mark.asyncio
    async def test_start_raises(self) -> None:
        handle = _ActorCellHandle(node_ids=["n0"])
        with pytest.raises(NotImplementedError, match="managed by the platform"):
            await handle.start()

    @pytest.mark.asyncio
    async def test_get_status_always_running(self) -> None:
        handle = _ActorCellHandle(node_ids=[])
        assert await handle.get_status() == "running"

    @pytest.mark.asyncio
    async def test_get_node_ids(self) -> None:
        handle = _ActorCellHandle(node_ids=["n0", "n1", "n2"])
        assert await handle.get_node_ids() == ["n0", "n1", "n2"]

    def test_cell_type_is_training(self) -> None:
        handle = _ActorCellHandle(node_ids=[])
        assert handle.cell_type == "actor"
        assert handle.cell_id == "actor"
