from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from miles.utils.mini_ft_controller import _CellSnapshot, _MiniFTController, _MiniFTControllerRunner


# ------------------------ helpers ------------------------


def _make_cell(
    *,
    name: str = "cell-0",
    healthy: bool = True,
) -> _CellSnapshot:
    return _CellSnapshot(name=name, healthy=healthy)


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_controller(
    *,
    get_cells: AsyncMock | None = None,
    suspend_cell: AsyncMock | None = None,
    resume_cell: AsyncMock | None = None,
    poll_interval: float = 0.01,
    resume_delay: float = 0.0,
    clock: _FakeClock | None = None,
) -> _MiniFTController:
    return _MiniFTController(
        get_cells=get_cells or AsyncMock(return_value=[]),
        suspend_cell=suspend_cell or AsyncMock(),
        resume_cell=resume_cell or AsyncMock(),
        poll_interval=poll_interval,
        resume_delay=resume_delay,
        clock=clock or _FakeClock(),
    )


def _run_controller_for_n_polls(
    controller: _MiniFTController,
    get_cells_mock: AsyncMock,
    responses: list[list[_CellSnapshot]],
) -> None:
    """Configure get_cells to return `responses` then stop the controller."""
    poll_count = 0

    async def _side_effect() -> list[_CellSnapshot]:
        nonlocal poll_count
        result = responses[poll_count] if poll_count < len(responses) else []
        poll_count += 1
        if poll_count >= len(responses):
            controller.request_stop()
        return result

    get_cells_mock.side_effect = _side_effect


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


# ------------------------ controller tests ------------------------


class TestControllerHealing:
    @pytest.mark.asyncio
    async def test_heal_on_unhealthy_cell(self) -> None:
        """Unhealthy cell triggers suspend → resume."""
        unhealthy_cell = _make_cell(name="cell-0", healthy=False)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_called_once_with("cell-0")
        resume_cell.assert_called_once_with("cell-0")

    @pytest.mark.asyncio
    async def test_skip_healthy_cell(self) -> None:
        """Healthy cell does not trigger heal."""
        healthy_cell = _make_cell(name="cell-0", healthy=True)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [[healthy_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_not_called()
        resume_cell.assert_not_called()

    @pytest.mark.asyncio
    async def test_heal_multiple_unhealthy_cells(self) -> None:
        """Multiple unhealthy cells are each healed."""
        cells = [
            _make_cell(name="cell-0", healthy=False),
            _make_cell(name="cell-1", healthy=False),
        ]
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [cells])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        assert suspend_cell.call_count == 2
        assert resume_cell.call_count == 2
        suspend_cell.assert_any_call("cell-0")
        suspend_cell.assert_any_call("cell-1")
        resume_cell.assert_any_call("cell-0")
        resume_cell.assert_any_call("cell-1")

    @pytest.mark.asyncio
    async def test_no_action_when_all_healthy(self) -> None:
        """All healthy → no suspend/resume calls."""
        cells = [
            _make_cell(name="cell-0", healthy=True),
            _make_cell(name="cell-1", healthy=True),
        ]
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [cells])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_not_called()
        resume_cell.assert_not_called()


class TestControllerBackoff:
    @pytest.mark.asyncio
    async def test_backoff_on_heal_failure(self) -> None:
        """Suspend raises → consecutive_failures increments, next_attempt_at increases."""
        unhealthy_cell = _make_cell(name="cell-0", healthy=False)
        clock = _FakeClock(start=100.0)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock(side_effect=RuntimeError("connection failed"))

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        backoff = controller._cell_backoffs["cell-0"]
        assert backoff.consecutive_failures == 1
        assert backoff.next_attempt_at == 110.0  # 100 + 5*2^1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        """Verify backoff delays: 5*2^1=10, 5*2^2=20, ..., capped at 300."""
        unhealthy_cell = _make_cell(name="cell-0", healthy=False)
        clock = _FakeClock(start=0.0)
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        get_cells = AsyncMock()
        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        expected_delays = [10, 20, 40, 80, 160, 300, 300]
        for expected_delay in expected_delays:
            # Advance clock past backoff window so the heal attempt is made
            backoff = controller._cell_backoffs.get("cell-0")
            if backoff:
                clock.now = backoff.next_attempt_at

            _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
            await asyncio.wait_for(controller.run(), timeout=5.0)

            backoff = controller._cell_backoffs["cell-0"]
            assert backoff.next_attempt_at == clock.now + expected_delay

    @pytest.mark.asyncio
    async def test_skips_heal_when_within_backoff_window(self) -> None:
        """Cell is not healed again until clock passes next_attempt_at."""
        unhealthy_cell = _make_cell(name="cell-0", healthy=False)
        clock = _FakeClock(start=0.0)
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        get_cells = AsyncMock()
        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        # Step 1: First poll → heal attempt fails, sets next_attempt_at
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert suspend_cell.call_count == 1

        # Step 2: Poll again without advancing clock → should skip heal
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert suspend_cell.call_count == 1  # no new call

        # Step 3: Advance clock past backoff → should attempt heal again
        backoff = controller._cell_backoffs["cell-0"]
        clock.now = backoff.next_attempt_at
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert suspend_cell.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_heal_resets_backoff(self) -> None:
        """Successful heal resets consecutive_failures to 0."""
        unhealthy_cell = _make_cell(name="cell-0", healthy=False)
        clock = _FakeClock(start=0.0)

        call_count = 0

        async def failing_then_succeeding_suspend(name: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")

        get_cells = AsyncMock()
        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=AsyncMock(side_effect=failing_then_succeeding_suspend),
            resume_cell=AsyncMock(),
            clock=clock,
        )

        # Step 1: First attempt fails
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)

        backoff = controller._cell_backoffs["cell-0"]
        assert backoff.consecutive_failures == 1

        # Step 2: Advance clock past backoff, second attempt succeeds
        clock.now = backoff.next_attempt_at
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)

        assert backoff.consecutive_failures == 0
        assert backoff.next_attempt_at == 0.0


class TestControllerLifecycle:
    @pytest.mark.asyncio
    async def test_poll_continues_after_get_cells_failure(self) -> None:
        """get_cells raises → controller does not exit, continues polling."""
        call_count = 0

        async def failing_then_stopping(controller_ref: list[_MiniFTController]) -> list[_CellSnapshot]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("network error")
            controller_ref[0].request_stop()
            return []

        controller = _make_controller()
        controller_ref = [controller]
        controller._get_cells = AsyncMock(
            side_effect=lambda: failing_then_stopping(controller_ref),
        )

        await asyncio.wait_for(controller.run(), timeout=5.0)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_request_stop_exits_loop(self) -> None:
        """call request_stop → run() returns."""
        get_cells = AsyncMock()
        controller = _make_controller(get_cells=get_cells)
        _run_controller_for_n_polls(controller, get_cells, [[]])

        await asyncio.wait_for(controller.run(), timeout=5.0)


# ------------------------ runner tests ------------------------


class TestRunnerGetCells:
    @pytest.mark.asyncio
    async def test_parses_healthy_and_unhealthy(self) -> None:
        """Parse CellList JSON into _CellSnapshot with correct healthy bool."""
        runner = _create_runner()

        cells_json = _build_cell_list_json(
            [
                _build_cell_json(name="actor-0", healthy_status="True"),
                _build_cell_json(name="actor-1", healthy_status="False"),
            ]
        )

        runner._client = AsyncMock()
        runner._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

        result = await runner._get_cells()

        assert len(result) == 2
        assert result[0] == _CellSnapshot(name="actor-0", healthy=True)
        assert result[1] == _CellSnapshot(name="actor-1", healthy=False)

    @pytest.mark.asyncio
    async def test_missing_healthy_condition_treated_as_unhealthy(self) -> None:
        """Cell with no Healthy condition → healthy=False."""
        runner = _create_runner()

        cell_json = _build_cell_json(name="actor-0")
        # Remove the Healthy condition, keep only Allocated
        cell_json["status"]["conditions"] = [{"type": "Allocated", "status": "True"}]

        cells_json = _build_cell_list_json([cell_json])
        runner._client = AsyncMock()
        runner._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

        result = await runner._get_cells()

        assert len(result) == 1
        assert result[0] == _CellSnapshot(name="actor-0", healthy=False)

    @pytest.mark.asyncio
    async def test_http_error_raises(self) -> None:
        """Verify HTTP 4xx/5xx propagated."""
        runner = _create_runner()
        runner._client = AsyncMock()
        runner._client.get = AsyncMock(return_value=_mock_response(status_code=500))

        with pytest.raises(httpx.HTTPStatusError):
            await runner._get_cells()


class TestRunnerPatchCell:
    @pytest.mark.asyncio
    async def test_suspend_sends_correct_patch(self) -> None:
        """Verify PATCH body for suspend uses CellPatch model."""
        runner = _create_runner()
        runner._client = AsyncMock()
        runner._client.patch = AsyncMock(return_value=_mock_response())

        await runner._suspend_cell("actor-0")

        runner._client.patch.assert_called_once()
        call_args = runner._client.patch.call_args
        assert call_args[0][0] == "/api/v1/cells/actor-0"
        body = json.loads(call_args[1]["content"])
        assert body == {"spec": {"suspend": True}}

    @pytest.mark.asyncio
    async def test_resume_sends_correct_patch(self) -> None:
        """Verify PATCH body for resume uses CellPatch model."""
        runner = _create_runner()
        runner._client = AsyncMock()
        runner._client.patch = AsyncMock(return_value=_mock_response())

        await runner._resume_cell("actor-0")

        runner._client.patch.assert_called_once()
        call_args = runner._client.patch.call_args
        assert call_args[0][0] == "/api/v1/cells/actor-0"
        body = json.loads(call_args[1]["content"])
        assert body == {"spec": {"suspend": False}}


class TestArgumentValidation:
    def test_requires_control_server_port(self) -> None:
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
