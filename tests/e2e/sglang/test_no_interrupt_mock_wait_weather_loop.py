"""E2E tests for no_interrupt strategy with mock wait weather agent.

Validates that the no_interrupt policy allows continuous multi-turn tool-call
agent loops to proceed without disruption from pause/continue perturbations.

These tests use MockSGLangServer (no real GPU required) and run the
mock_wait_weather_agent against it while a background task periodically
triggers pause_generation/continue_generation to simulate weight-update windows.

Pass criteria (from plan section 5.D4):
- Continuous multi-turn (>=12) tool-call + wait cycles complete
- Multiple pause/continue perturbations mid-loop don't break the agent
- For no_interrupt modes, agent never sees finish_reason=abort
"""

import asyncio
import logging
import re

import pytest
import requests
from tests.e2e.sglang.utils.mock_wait_weather_agent import CITIES, run_mock_wait_weather_loop

from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult

logger = logging.getLogger(__name__)

MAX_TURNS = 12


def _weather_tool_call_process_fn(prompt: str) -> ProcessResult:
    """Always return a get_weather tool call for the next city in rotation."""
    # Count how many tool_response blocks appear to determine which city to query next
    tool_response_count = len(re.findall(r"<tool_response>", prompt))
    city_idx = tool_response_count % len(CITIES)
    city = CITIES[city_idx]

    text = (
        f"Let me check the weather in {city}.\n"
        "<tool_call>\n"
        f'{{"name": "get_weather", "arguments": {{"location": "{city}"}}}}\n'
        "</tool_call>"
    )
    return ProcessResult(text=text, finish_reason="stop")


async def _run_perturbation_loop(
    server_url: str,
    *,
    pause_mode: str = "retract",
    interval: float = 0.5,
    count: int = 5,
):
    """Background task that periodically pauses and continues generation."""
    pauses_done = 0
    for _ in range(count):
        await asyncio.sleep(interval)
        try:
            resp = requests.post(
                f"{server_url}/pause_generation",
                json={"mode": pause_mode},
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Perturbation pause failed: %s", e)
            continue

        await asyncio.sleep(0.1)  # brief pause window

        try:
            resp = requests.post(
                f"{server_url}/continue_generation",
                json={},
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Perturbation continue failed: %s", e)
            continue

        pauses_done += 1
        logger.info("Perturbation cycle %d/%d done (mode=%s)", pauses_done, count, pause_mode)

    return pauses_done


async def _run_test(server_url: str, pause_mode: str):
    """Run agent + perturbation concurrently, return (result, pauses_done)."""
    agent_task = asyncio.create_task(
        run_mock_wait_weather_loop(
            server_url,
            max_turns=MAX_TURNS,
            wait_s=0.1,
        )
    )
    perturb_task = asyncio.create_task(
        _run_perturbation_loop(
            server_url,
            pause_mode=pause_mode,
            interval=0.3,
            count=5,
        )
    )
    return await asyncio.gather(agent_task, perturb_task)


@pytest.fixture
def mock_weather_server():
    """Start a MockSGLangServer with weather tool-call process_fn."""
    port = find_available_port(30000)
    server = MockSGLangServer(
        model_name="Qwen/Qwen3-0.6B",
        process_fn=_weather_tool_call_process_fn,
        host="127.0.0.1",
        port=port,
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_mock_wait_weather_loop_no_interrupt_retract(mock_weather_server):
    """no_interrupt + retract: agent completes all turns despite pause/continue perturbations."""
    server = mock_weather_server

    result, pauses_done = asyncio.run(_run_test(server.url, "retract"))

    assert result["turns_completed"] == MAX_TURNS, f"Expected {MAX_TURNS} turns, got {result}"
    assert result["total_tool_calls"] >= MAX_TURNS
    assert len(result["errors"]) == 0, f"Errors: {result['errors']}"
    assert result["aborted_seen"] == 0, "no_interrupt should not expose abort to agent"

    assert server.pause_calls >= 1, "Perturbation should have triggered at least 1 pause"
    assert server.continue_calls >= 1
    assert server.last_pause_mode == "retract"


def test_mock_wait_weather_loop_no_interrupt_in_place(mock_weather_server):
    """no_interrupt + in_place: same as retract but with in_place pause mode."""
    server = mock_weather_server

    result, pauses_done = asyncio.run(_run_test(server.url, "in_place"))

    assert result["turns_completed"] == MAX_TURNS, f"Expected {MAX_TURNS} turns, got {result}"
    assert result["total_tool_calls"] >= MAX_TURNS
    assert len(result["errors"]) == 0, f"Errors: {result['errors']}"
    assert result["aborted_seen"] == 0

    assert server.pause_calls >= 1
    assert server.continue_calls >= 1
    assert server.last_pause_mode == "in_place"


def test_mock_wait_weather_loop_legacy_abort_resume_control(mock_weather_server):
    """legacy_abort_resume control group: agent completes via abort/resume path."""
    server = mock_weather_server

    result, pauses_done = asyncio.run(_run_test(server.url, "abort"))

    # Legacy mode should still complete (may be slower due to abort/resume)
    assert result["turns_completed"] == MAX_TURNS, f"Expected {MAX_TURNS} turns, got {result}"
    assert result["total_tool_calls"] >= MAX_TURNS
    assert len(result["errors"]) == 0, f"Errors: {result['errors']}"

    assert server.pause_calls >= 1
    assert server.last_pause_mode == "abort"
