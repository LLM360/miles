import pytest
import httpx

from miles.utils.control_server_utils import SubsystemRegistry, _StopRequest, _create_control_app


class _MockHandle:
    def __init__(
        self,
        subsystem_id: str,
        subsystem_type: str,
        status: str = "running",
        node_ids: list[str] | None = None,
        stop_error: Exception | None = None,
        start_error: Exception | None = None,
    ) -> None:
        self.subsystem_id = subsystem_id
        self.subsystem_type = subsystem_type
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


@pytest.fixture
def registry() -> SubsystemRegistry:
    return SubsystemRegistry()


@pytest.fixture
def async_client(registry: SubsystemRegistry) -> httpx.AsyncClient:
    app = _create_control_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ── SubsystemRegistry tests ──


def test_register_and_get_by_id(registry: SubsystemRegistry) -> None:
    handle = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout")
    registry.register(handle)
    assert registry.get("cell-0") is handle


def test_get_unknown_id_raises_key_error(registry: SubsystemRegistry) -> None:
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_get_all_returns_all_registered(registry: SubsystemRegistry) -> None:
    h1 = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout")
    h2 = _MockHandle(subsystem_id="cell-1", subsystem_type="rollout")
    registry.register(h1)
    registry.register(h2)

    all_handles = registry.get_all()
    assert len(all_handles) == 2
    assert h1 in all_handles
    assert h2 in all_handles


def test_register_duplicate_id_raises(registry: SubsystemRegistry) -> None:
    h1 = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout")
    h2 = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout")
    registry.register(h1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(h2)


# ── GET /subsystems tests ──


@pytest.mark.asyncio
async def test_get_subsystems_empty_registry(async_client: httpx.AsyncClient) -> None:
    resp = await async_client.get("/subsystems")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_subsystems_returns_all(registry: SubsystemRegistry, async_client: httpx.AsyncClient) -> None:
    registry.register(_MockHandle(
        subsystem_id="training",
        subsystem_type="training",
        status="running",
        node_ids=["node-0", "node-1"],
    ))
    registry.register(_MockHandle(
        subsystem_id="cell-0",
        subsystem_type="rollout",
        status="running",
        node_ids=["node-2"],
    ))
    registry.register(_MockHandle(
        subsystem_id="cell-1",
        subsystem_type="rollout",
        status="stopped",
        node_ids=["node-3"],
    ))

    resp = await async_client.get("/subsystems")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3

    by_id = {d["subsystem_id"]: d for d in data}
    assert by_id["training"]["subsystem_type"] == "training"
    assert by_id["training"]["status"] == "running"
    assert by_id["training"]["node_ids"] == ["node-0", "node-1"]
    assert by_id["cell-0"]["subsystem_type"] == "rollout"
    assert by_id["cell-0"]["status"] == "running"
    assert by_id["cell-1"]["status"] == "stopped"


@pytest.mark.asyncio
async def test_get_subsystems_reflects_status_change(
    registry: SubsystemRegistry, async_client: httpx.AsyncClient
) -> None:
    handle = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout", status="running")
    registry.register(handle)

    resp1 = await async_client.get("/subsystems")
    assert resp1.json()[0]["status"] == "running"

    handle._status = "stopped"

    resp2 = await async_client.get("/subsystems")
    assert resp2.json()[0]["status"] == "stopped"


# ── POST /subsystems/{subsystem_id}/stop tests ──


@pytest.mark.asyncio
async def test_stop_calls_handle_with_default_timeout(
    registry: SubsystemRegistry, async_client: httpx.AsyncClient
) -> None:
    handle = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout")
    registry.register(handle)

    resp = await async_client.post("/subsystems/cell-0/stop")
    assert resp.status_code == 200
    assert handle.stop_calls == [30]


@pytest.mark.asyncio
async def test_stop_calls_handle_with_custom_timeout(
    registry: SubsystemRegistry, async_client: httpx.AsyncClient
) -> None:
    handle = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout")
    registry.register(handle)

    resp = await async_client.post("/subsystems/cell-0/stop", json={"timeout_seconds": 60})
    assert resp.status_code == 200
    assert handle.stop_calls == [60]


@pytest.mark.asyncio
async def test_stop_unknown_subsystem_returns_404(async_client: httpx.AsyncClient) -> None:
    resp = await async_client.post("/subsystems/nonexistent/stop")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stop_already_stopped_is_idempotent(
    registry: SubsystemRegistry, async_client: httpx.AsyncClient
) -> None:
    handle = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout", status="stopped")
    registry.register(handle)

    resp = await async_client.post("/subsystems/cell-0/stop")
    assert resp.status_code == 200
    assert len(handle.stop_calls) == 1


# ── POST /subsystems/{subsystem_id}/start tests ──


@pytest.mark.asyncio
async def test_start_calls_handle(registry: SubsystemRegistry, async_client: httpx.AsyncClient) -> None:
    handle = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout", status="stopped")
    registry.register(handle)

    resp = await async_client.post("/subsystems/cell-0/start")
    assert resp.status_code == 200
    assert handle.start_calls == 1


@pytest.mark.asyncio
async def test_start_unknown_subsystem_returns_404(async_client: httpx.AsyncClient) -> None:
    resp = await async_client.post("/subsystems/nonexistent/start")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_start_already_running_is_idempotent(
    registry: SubsystemRegistry, async_client: httpx.AsyncClient
) -> None:
    handle = _MockHandle(subsystem_id="cell-0", subsystem_type="rollout", status="running")
    registry.register(handle)

    resp = await async_client.post("/subsystems/cell-0/start")
    assert resp.status_code == 200
    assert handle.start_calls == 1


# ── Error handling tests ──


@pytest.mark.asyncio
async def test_stop_handle_raises_returns_500(
    registry: SubsystemRegistry, async_client: httpx.AsyncClient
) -> None:
    handle = _MockHandle(
        subsystem_id="cell-0",
        subsystem_type="rollout",
        stop_error=RuntimeError("engine crashed"),
    )
    registry.register(handle)

    resp = await async_client.post("/subsystems/cell-0/stop")
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_start_handle_raises_returns_500(
    registry: SubsystemRegistry, async_client: httpx.AsyncClient
) -> None:
    handle = _MockHandle(
        subsystem_id="cell-0",
        subsystem_type="rollout",
        start_error=RuntimeError("engine crashed"),
    )
    registry.register(handle)

    resp = await async_client.post("/subsystems/cell-0/start")
    assert resp.status_code == 500


# ── _RolloutSubsystemHandle tests ──


class _MockRemoteCall:
    """Simulates `actor.method.remote(arg)` returning a value that `ray.get` resolves."""

    def __init__(self, return_value: object) -> None:
        self._return_value = return_value

    def remote(self, *args: object, **kwargs: object) -> object:
        return self._return_value


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
        self.stopped_calls: list[str] = []


@pytest.mark.asyncio
async def test_rollout_handle_stop_delegates_to_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    import miles.utils.control_server_utils as mod

    manager = _MockRolloutManager()
    monkeypatch.setattr(mod.ray, "get", lambda ref: ref)

    handle = mod._RolloutSubsystemHandle(rollout_manager=manager, cell_id="cell-0", node_ids=["n0"])
    await handle.stop(timeout_seconds=45)


@pytest.mark.asyncio
async def test_rollout_handle_start_delegates_to_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    import miles.utils.control_server_utils as mod

    manager = _MockRolloutManager()
    monkeypatch.setattr(mod.ray, "get", lambda ref: ref)

    handle = mod._RolloutSubsystemHandle(rollout_manager=manager, cell_id="cell-0", node_ids=["n0"])
    await handle.start()


@pytest.mark.asyncio
async def test_rollout_handle_get_status_delegates_to_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    import miles.utils.control_server_utils as mod

    manager = _MockRolloutManager(status_return="stopped")
    monkeypatch.setattr(mod.ray, "get", lambda ref: ref)

    handle = mod._RolloutSubsystemHandle(rollout_manager=manager, cell_id="cell-0", node_ids=["n0"])
    status = await handle.get_status()
    assert status == "stopped"


@pytest.mark.asyncio
async def test_rollout_handle_get_node_ids() -> None:
    from miles.utils.control_server_utils import _RolloutSubsystemHandle

    handle = _RolloutSubsystemHandle(rollout_manager=object(), cell_id="cell-0", node_ids=["n0", "n1"])
    assert await handle.get_node_ids() == ["n0", "n1"]


def test_rollout_handle_subsystem_type_is_rollout() -> None:
    from miles.utils.control_server_utils import _RolloutSubsystemHandle

    handle = _RolloutSubsystemHandle(rollout_manager=object(), cell_id="cell-0", node_ids=[])
    assert handle.subsystem_type == "rollout"
    assert handle.subsystem_id == "cell-0"
