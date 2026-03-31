"""Mock wait weather agent for no_interrupt E2E testing.

Runs a fixed loop of weather tool-calls with configurable waits between turns.
Designed to exercise the no_interrupt pause/continue path under perturbation.

The agent:
- Sends /v1/chat/completions each turn
- Expects assistant to return tool_calls (get_weather)
- Executes mock weather tool, writes back tool message
- Sleeps wait_s between turns to simulate wait agent
- Loops for max_turns (does not depend on final_answer exit)
- Returns metadata: turns_completed, total_tool_calls, wait_calls, aborted_seen
"""

import asyncio
import json
import logging

import httpx

logger = logging.getLogger(__name__)

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. Beijing",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

CITIES = ["Beijing", "Shanghai", "Tokyo", "New York", "London", "Paris"]

MOCK_WEATHER_RESULTS = {
    "Beijing": {"temperature_celsius": 22, "condition": "sunny", "humidity": 45},
    "Shanghai": {"temperature_celsius": 28, "condition": "cloudy", "humidity": 70},
    "Tokyo": {"temperature_celsius": 18, "condition": "rainy", "humidity": 80},
    "New York": {"temperature_celsius": 12, "condition": "windy", "humidity": 55},
    "London": {"temperature_celsius": 8, "condition": "foggy", "humidity": 90},
    "Paris": {"temperature_celsius": 15, "condition": "clear", "humidity": 50},
}

DEFAULT_WEATHER = {"temperature_celsius": 20, "condition": "unknown", "humidity": 60}


def mock_weather_tool(location: str) -> str:
    result = MOCK_WEATHER_RESULTS.get(location, DEFAULT_WEATHER)
    return json.dumps(result)


async def run_mock_wait_weather_loop(
    base_url: str,
    *,
    max_turns: int = 12,
    wait_s: float = 0.2,
    timeout: float = 60.0,
) -> dict:
    """Run a continuous weather-query loop against base_url.

    Returns metadata dict with:
      turns_completed, total_tool_calls, wait_calls, aborted_seen, errors
    """
    turns_completed = 0
    total_tool_calls = 0
    wait_calls = 0
    aborted_seen = 0
    errors = []

    messages = [
        {
            "role": "user",
            "content": (
                "You are a weather reporter. Each turn, call get_weather for a city "
                "and report the result. Query cities in order: "
                + ", ".join(CITIES)
                + ". Repeat from the beginning when you run out of cities."
            ),
        }
    ]

    async with httpx.AsyncClient(timeout=timeout) as client:
        for turn in range(1, max_turns + 1):
            # wait between turns
            if turn > 1:
                await asyncio.sleep(wait_s)
                wait_calls += 1

            payload = {
                "messages": messages,
                "tools": WEATHER_TOOLS,
                "tool_choice": "auto",
            }

            try:
                resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            except Exception as e:
                errors.append(f"Turn {turn}: request failed: {e}")
                logger.error("Turn %d: request exception: %s", turn, e)
                break

            if resp.status_code != 200:
                errors.append(f"Turn {turn}: HTTP {resp.status_code}")
                logger.error("Turn %d: non-200 response: %d %s", turn, resp.status_code, resp.text[:200])
                break

            data = resp.json()
            choice = data["choices"][0]
            assistant_msg = choice["message"]
            finish_reason = choice.get("finish_reason", "")

            if finish_reason == "abort":
                aborted_seen += 1
                logger.warning("Turn %d: finish_reason=abort detected", turn)

            messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                errors.append(f"Turn {turn}: no tool_calls in response")
                logger.error("Turn %d: expected tool_calls but got none", turn)
                break

            for tc in tool_calls:
                raw_args = tc["function"].get("arguments", "{}")
                try:
                    params = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    params = {}
                location = params.get("location", CITIES[(total_tool_calls) % len(CITIES)])
                result = mock_weather_tool(location)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
                total_tool_calls += 1

            turns_completed = turn
            logger.info(
                "Turn %d: %d tool_calls, total=%d",
                turn,
                len(tool_calls),
                total_tool_calls,
            )

    return {
        "turns_completed": turns_completed,
        "total_tool_calls": total_tool_calls,
        "wait_calls": wait_calls,
        "aborted_seen": aborted_seen,
        "errors": errors,
    }
