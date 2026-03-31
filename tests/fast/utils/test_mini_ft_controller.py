from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from miles.utils.mini_ft_controller import CellSnapshot, _MiniFTController, _MiniFTControllerRunner


# ------------------------ helpers: controller tests ------------------------


def _make_cell(
    *,
    name: str = "cell-0",
    healthy_status: str = "True",
    healthy_reason: str | None = None,
) -> CellSnapshot:
    return CellSnapshot(name=name, healthy_status=healthy_status, healthy_reason=healthy_reason)


def _make_controller(
    *,
    get_cells: AsyncMock | None = None,
    suspend_cell: AsyncMock | None = None,
    resume_cell: AsyncMock | None = None,
    poll_interval: float = 0.01,
    resume_delay: float = 0.0,
) -> _MiniFTController:
    return _MiniFTController(
        get_cells=get_cells or AsyncMock(return_value=[]),
        suspend_cell=suspend_cell or AsyncMock(),
        resume_cell=resume_cell or AsyncMock(),
        poll_interval=poll_interval,
        resume_delay=resume_delay,
    )


# ------------------------ controller tests ------------------------


@pytest.mark.asyncio
async def test_heal_on_fatal_error() -> None:
    """Fatal cell triggers suspend → sleep → resume in order."""
    fatal_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal")
    get_cells = AsyncMock(return_value=[fatal_cell])
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=get_cells,
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )

    get_cells.side_effect = [
        [fatal_cell],
        asyncio.CancelledError(),
    ]

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(controller.run(), timeout=5.0)

    suspend_cell.assert_called_once_with("cell-0")
    resume_cell.assert_called_once_with("cell-0")


@pytest.mark.asyncio
async def test_skip_degraded_cell() -> None:
    """Degraded cell does not trigger heal."""
    degraded_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Degraded")
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[degraded_cell]),
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )
    await controller._poll_and_heal()

    suspend_cell.assert_not_called()
    resume_cell.assert_not_called()


@pytest.mark.asyncio
async def test_skip_healthy_cell() -> None:
    """Healthy=True cell does not trigger heal."""
    healthy_cell = _make_cell(name="cell-0", healthy_status="True")
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[healthy_cell]),
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )
    await controller._poll_and_heal()

    suspend_cell.assert_not_called()
    resume_cell.assert_not_called()


@pytest.mark.asyncio
async def test_heal_multiple_fatal_cells() -> None:
    """Multiple Fatal cells are each healed."""
    cells = [
        _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal"),
        _make_cell(name="cell-1", healthy_status="False", healthy_reason="Fatal"),
    ]
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=AsyncMock(return_value=cells),
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )
    await controller._poll_and_heal()

    assert suspend_cell.call_count == 2
    assert resume_cell.call_count == 2
    suspend_cell.assert_any_call("cell-0")
    suspend_cell.assert_any_call("cell-1")
    resume_cell.assert_any_call("cell-0")
    resume_cell.assert_any_call("cell-1")


@pytest.mark.asyncio
async def test_backoff_on_heal_failure() -> None:
    """Suspend raises → consecutive_failures increments, next_attempt_at increases."""
    fatal_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal")
    suspend_cell = AsyncMock(side_effect=RuntimeError("connection failed"))

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[fatal_cell]),
        suspend_cell=suspend_cell,
    )

    await controller._poll_and_heal()

    backoff = controller._cell_backoffs["cell-0"]
    assert backoff.consecutive_failures == 1
    assert backoff.next_attempt_at > 0


@pytest.mark.asyncio
async def test_exponential_backoff_timing() -> None:
    """Verify backoff delays: 5*2^1=10, 5*2^2=20, 5*2^3=40, ..., capped at 300."""
    fatal_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal")
    suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[fatal_cell]),
        suspend_cell=suspend_cell,
    )

    expected_delays = [10, 20, 40, 80, 160, 300, 300]
    for expected_delay in expected_delays:
        backoff = controller._cell_backoffs.get("cell-0")
        if backoff:
            backoff.next_attempt_at = 0.0

        before = time.monotonic()
        await controller._poll_and_heal()
        after = time.monotonic()

        backoff = controller._cell_backoffs["cell-0"]
        actual_delay = backoff.next_attempt_at - after
        assert abs(actual_delay - expected_delay) < 2.0, (
            f"Expected delay ~{expected_delay}, got {actual_delay:.1f}"
        )


@pytest.mark.asyncio
async def test_successful_heal_resets_backoff() -> None:
    """Successful heal resets consecutive_failures to 0."""
    fatal_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal")

    call_count = 0

    async def failing_then_succeeding_suspend(name: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("fail")

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[fatal_cell]),
        suspend_cell=AsyncMock(side_effect=failing_then_succeeding_suspend),
        resume_cell=AsyncMock(),
    )

    await controller._poll_and_heal()
    backoff = controller._cell_backoffs["cell-0"]
    assert backoff.consecutive_failures == 1

    backoff.next_attempt_at = 0.0
    await controller._poll_and_heal()
    assert backoff.consecutive_failures == 0
    assert backoff.next_attempt_at == 0.0


@pytest.mark.asyncio
async def test_poll_continues_after_get_cells_failure() -> None:
    """get_cells raises → controller does not exit."""
    call_count = 0

    async def failing_get_cells() -> list[CellSnapshot]:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RuntimeError("network error")
        raise asyncio.CancelledError()

    controller = _make_controller(
        get_cells=AsyncMock(side_effect=failing_get_cells),
    )

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(controller.run(), timeout=5.0)

    assert call_count == 3


@pytest.mark.asyncio
async def test_request_stop_exits_loop() -> None:
    """call request_stop → run() returns."""
    controller = _make_controller()

    async def stop_after_first_poll() -> list[CellSnapshot]:
        controller.request_stop()
        return []

    controller._get_cells = AsyncMock(side_effect=stop_after_first_poll)

    await asyncio.wait_for(controller.run(), timeout=5.0)


@pytest.mark.asyncio
async def test_no_action_when_all_healthy() -> None:
    """All Healthy=True → no suspend/resume calls."""
    cells = [
        _make_cell(name="cell-0", healthy_status="True"),
        _make_cell(name="cell-1", healthy_status="True"),
    ]
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=AsyncMock(return_value=cells),
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )
    await controller._poll_and_heal()

    suspend_cell.assert_not_called()
    resume_cell.assert_not_called()


# ------------------------ helpers: runner tests ------------------------


def _build_cell_json(
    *,
    name: str = "actor-0",
    healthy_status: str = "True",
    healthy_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "apiVersion": "miles.io/v1",
        "kind": "Cell",
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": "actor", "miles.io/cell-index": "0"},
        },
        "spec": {"suspend": False},
        "status": {
            "phase": "Running",
            "conditions": [
                {"type": "Allocated", "status": "True"},
                {"type": "Healthy", "status": healthy_status, "reason": healthy_reason},
            ],
        },
    }


def _build_cell_list_json(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {"apiVersion": "miles.io/v1", "kind": "CellList", "items": cells}


def _create_runner() -> _MiniFTControllerRunner:
    return _MiniFTControllerRunner(
        control_server_url="http://127.0.0.1:8080",
        poll_interval=10.0,
        resume_delay=5.0,
    )


def _mock_response(*, status_code: int = 200, json_data: Any = None) -> httpx.Response:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=response,
        )
    return response


# ------------------------ runner tests ------------------------


@pytest.mark.asyncio
async def test_get_cells_parses_response() -> None:
    """Mock httpx returning CellList JSON → verify CellSnapshot list correct."""
    runner = _create_runner()

    cells_json = _build_cell_list_json([
        _build_cell_json(name="actor-0", healthy_status="True"),
        _build_cell_json(name="actor-1", healthy_status="False", healthy_reason="Fatal"),
    ])

    runner._client = AsyncMock()
    runner._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

    result = await runner._get_cells()

    assert len(result) == 2
    assert result[0] == CellSnapshot(name="actor-0", healthy_status="True", healthy_reason=None)
    assert result[1] == CellSnapshot(name="actor-1", healthy_status="False", healthy_reason="Fatal")


@pytest.mark.asyncio
async def test_get_cells_extracts_healthy_condition() -> None:
    """Verify Healthy condition status and reason correctly extracted."""
    runner = _create_runner()

    cells_json = _build_cell_list_json([
        _build_cell_json(name="cell-0", healthy_status="False", healthy_reason="Fatal"),
    ])

    runner._client = AsyncMock()
    runner._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

    result = await runner._get_cells()

    assert result[0].healthy_status == "False"
    assert result[0].healthy_reason == "Fatal"


@pytest.mark.asyncio
async def test_suspend_cell_sends_correct_patch() -> None:
    """Verify PATCH body for suspend."""
    runner = _create_runner()
    runner._client = AsyncMock()
    runner._client.patch = AsyncMock(return_value=_mock_response())

    await runner._suspend_cell("actor-0")

    runner._client.patch.assert_called_once_with(
        "/api/v1/cells/actor-0",
        json={"spec": {"suspend": True}},
    )


@pytest.mark.asyncio
async def test_resume_cell_sends_correct_patch() -> None:
    """Verify PATCH body for resume."""
    runner = _create_runner()
    runner._client = AsyncMock()
    runner._client.patch = AsyncMock(return_value=_mock_response())

    await runner._resume_cell("actor-0")

    runner._client.patch.assert_called_once_with(
        "/api/v1/cells/actor-0",
        json={"spec": {"suspend": False}},
    )


@pytest.mark.asyncio
async def test_get_cells_http_error_raises() -> None:
    """Verify HTTP 4xx/5xx propagated."""
    runner = _create_runner()
    runner._client = AsyncMock()
    runner._client.get = AsyncMock(return_value=_mock_response(status_code=500))

    with pytest.raises(httpx.HTTPStatusError):
        await runner._get_cells()


def test_argument_validation_requires_control_server_port() -> None:
    """mini_ft_controller_enable=True + control_server_port=0 → error."""
    from miles.utils.arguments import miles_validate_args

    args = argparse.Namespace(
        mini_ft_controller_enable=True,
        control_server_port=0,
        use_fault_tolerance=False,
        ft_components=None,
        eval_datasets=None,
        eval_data=None,
    )

    with pytest.raises(ValueError, match="--mini-ft-controller-enable requires --control-server-port"):
        miles_validate_args(args)
