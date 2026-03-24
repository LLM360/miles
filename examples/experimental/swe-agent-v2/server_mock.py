"""
Mock agent server for pipeline testing.

Simulates the real Harbor-based server.py by making LLM calls through the
Miles Router (so TITO records are generated) without requiring Harbor,
Docker, or any real agent environment.

The mock:
  1. Receives a RunRequest with base_url pointing to the Miles Router session
  2. Makes one or more /v1/chat/completions calls through that URL
     (this is the critical part — without these calls, the Router records
     nothing and the sample is marked ABORTED)
  3. Returns a RunResponse with a configurable reward

Usage:
    python server_mock.py --port 11000

    # Fixed reward of 1.0, single turn:
    python server_mock.py --port 11000 --reward 1.0 --turns 1

    # Random reward in [0, 1], 3 turns per instance:
    python server_mock.py --port 11000 --reward random --turns 3
"""

import argparse
import asyncio
import logging
import os
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None
_config: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _semaphore
    max_concurrent = int(os.getenv("AGENT_MAX_CONCURRENT", "8"))
    _semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Mock server ready — max_concurrent={max_concurrent}, config={_config}")
    yield


app = FastAPI(title="Mock Agent Server", lifespan=_lifespan)


class RunRequest(BaseModel):
    base_url: str
    model: str
    sampling_params: dict[str, Any] = {}
    api_key: str = "dummy"
    instance_id: str = ""
    agent_name: str = ""

    model_config = {"extra": "allow"}


class RunResponse(BaseModel):
    reward: float = 0.0
    exit_status: str = ""
    agent_metrics: dict[str, Any] = {}
    eval_report: dict[str, Any] = {}


MOCK_SYSTEM_PROMPT = "You are a helpful assistant working on a software task. Think step by step."

MOCK_TOOL_OBSERVATION = (
    "File edited successfully. The test suite now passes with 12/12 tests green."
)


def _build_reward() -> float:
    mode = _config.get("reward", "random")
    if mode == "random":
        return float(random.randint(0, 1))
    return float(mode)


async def _make_llm_call(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    messages: list[dict],
    api_key: str,
    sampling_params: dict[str, Any],
) -> dict | None:
    """Make a single /v1/chat/completions call through the Miles Router."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": sampling_params.get("max_tokens", 1024),
    }
    for key in ("temperature", "top_p"):
        if key in sampling_params:
            payload[key] = sampling_params[key]

    try:
        resp = await client.post(
            f"{url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


async def _run_mock_agent(request: RunRequest) -> dict[str, Any]:
    """Simulate an agent by making LLM calls through the Miles Router."""
    num_turns = int(_config.get("turns", 1))
    timeout = float(_config.get("timeout", 300))

    prompt = getattr(request, "problem_statement", None) or getattr(request, "instruction", None) or ""
    if not prompt:
        extra = request.model_extra or {}
        prompt = extra.get("problem_statement") or extra.get("instruction") or extra.get("prompt", "")

    if not prompt:
        prompt = f"You are working on task {request.instance_id}. Describe what you would do."

    messages: list[dict[str, str]] = [
        {"role": "system", "content": MOCK_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    async with httpx.AsyncClient(timeout=timeout) as client:
        for turn in range(num_turns):
            result = await _make_llm_call(
                client,
                request.base_url,
                request.model,
                messages,
                request.api_key,
                request.sampling_params,
            )
            if result is None:
                return {
                    "reward": 0.0,
                    "exit_status": "AgentError",
                    "agent_metrics": {"turns": turn},
                    "eval_report": {},
                }

            assistant_msg = result["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": assistant_msg})

            if turn < num_turns - 1:
                messages.append({"role": "user", "content": MOCK_TOOL_OBSERVATION})

    reward = _build_reward()
    return {
        "reward": reward,
        "exit_status": "Submitted",
        "agent_metrics": {"turns": num_turns},
        "eval_report": {"resolved": reward > 0.5},
    }


@app.post("/run")
async def run_instance(request: RunRequest) -> RunResponse:
    logger.info(f"Mock run: instance_id={request.instance_id}")
    assert _semaphore is not None
    async with _semaphore:
        result = await _run_mock_agent(request)
    logger.info(
        f"Mock done: instance_id={request.instance_id}, "
        f"reward={result['reward']}, exit_status={result['exit_status']}"
    )
    return RunResponse(**result)


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    global _config

    parser = argparse.ArgumentParser(description="Mock Agent Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument(
        "--reward",
        type=str,
        default="random",
        help="Reward mode: 'random' for uniform [0,1], or a fixed float (default: random)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=1,
        help="Number of LLM turns per instance (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="HTTP timeout in seconds for LLM calls (default: 300)",
    )
    args = parser.parse_args()

    os.environ["AGENT_MAX_CONCURRENT"] = str(args.max_concurrent)
    _config = {"reward": args.reward, "turns": args.turns, "timeout": args.timeout}

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
