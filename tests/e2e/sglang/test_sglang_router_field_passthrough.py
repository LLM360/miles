"""E2E test: verify the sglang Rust router preserves unknown request fields.

Tests the fix from https://github.com/sgl-project/sglang/pull/21571

The default ``--backend sglang`` HTTP router deserializes requests into a typed
Rust struct and re-serializes before forwarding.  Without the ``#[serde(flatten)]``
fix, any fields not defined in the Rust ``ChatCompletionRequest`` struct are
silently dropped.

This test starts:
  1. A bare sglang server (worker)
  2. The ``sglang-router`` binary in front of it with ``--backend sglang``

It then sends the **same** ``/v1/chat/completions`` request both directly to
the worker and through the router, and asserts that key extra fields
(``return_meta_info``, ``return_prompt_token_ids``) survive the round-trip
through the router.

Requires 1 GPU.
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
import requests as http_requests
from tests.e2e.sglang.utils.sglang_server import start_sglang_server

from miles.utils.http_utils import find_available_port

MODEL_PATH = os.environ.get("SGLANG_E2E_MODEL_PATH", "Qwen/Qwen3-0.6B")
SEED = 42
MAX_NEW_TOKENS = 20
TIMEOUT_SECS = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_sglang_router(
    worker_url: str,
    host: str = "127.0.0.1",
    port: int | None = None,
    backend: str = "sglang",
    startup_timeout_secs: float = 30.0,
) -> tuple[subprocess.Popen, str, Path]:
    """Start the ``sglang-router`` binary and wait until it is healthy."""
    if port is None:
        port = find_available_port(35000)

    log_path = Path(f"/tmp/sglang_router_e2e_{port}.log")
    log_file = log_path.open("w", encoding="utf-8")

    cmd = [
        "sglang-router",
        "launch",
        "--host",
        host,
        "--port",
        str(port),
        "--worker-urls",
        worker_url,
        "--backend",
        backend,
        "--policy",
        "round_robin",
        "--disable-health-check",
    ]

    env = os.environ.copy()
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    router_url = f"http://{host}:{port}"
    deadline = time.monotonic() + startup_timeout_secs
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_tail = log_path.read_text(errors="ignore")[-2000:]
            raise RuntimeError(f"sglang-router exited early (code={process.returncode}). Log:\n{log_tail}")
        try:
            r = http_requests.get(f"{router_url}/health", timeout=3)
            if r.status_code == 200:
                return process, router_url, log_path
            last_error = f"status={r.status_code}"
        except http_requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)

    log_tail = log_path.read_text(errors="ignore")[-2000:]
    process.kill()
    raise TimeoutError(f"sglang-router did not become healthy. Last: {last_error}. Log:\n{log_tail}")


def _stop_router(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _build_payload(extra_fields: dict | None = None) -> dict:
    payload = {
        "model": MODEL_PATH,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Say hello in one word."},
        ],
        "temperature": 0,
        "max_completion_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sglang_server():
    server = start_sglang_server(model_path=MODEL_PATH)
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="module")
def sglang_router(sglang_server):
    process, router_url, log_path = _start_sglang_router(
        worker_url=sglang_server.base_url,
        backend="sglang",
    )
    try:
        yield router_url
    finally:
        _stop_router(process)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.system
def test_return_meta_info_passthrough(sglang_server, sglang_router):
    """``return_meta_info=True`` must survive the router and produce ``meta_info``
    in the response choices."""
    extra = {"return_meta_info": True, "logprobs": True}
    payload = _build_payload(extra)

    # Direct to worker
    direct_resp = http_requests.post(
        f"{sglang_server.base_url}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT_SECS,
    )
    assert direct_resp.status_code == 200, f"direct request failed: {direct_resp.text}"
    direct_choice = direct_resp.json()["choices"][0]
    assert "meta_info" in direct_choice, (
        "Sanity check failed: direct request missing meta_info. " "Is the sglang server version correct?"
    )

    # Through router
    router_resp = http_requests.post(
        f"{sglang_router}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT_SECS,
    )
    assert router_resp.status_code == 200, f"router request failed: {router_resp.text}"
    router_choice = router_resp.json()["choices"][0]
    assert "meta_info" in router_choice, (
        "meta_info missing in router response — the sglang router dropped "
        "the return_meta_info field during request forwarding"
    )


@pytest.mark.system
def test_return_prompt_token_ids_passthrough(sglang_server, sglang_router):
    """``return_prompt_token_ids=True`` must survive the router and produce
    ``prompt_token_ids`` in the response choices."""
    extra = {"return_prompt_token_ids": True}
    payload = _build_payload(extra)

    # Direct
    direct_resp = http_requests.post(
        f"{sglang_server.base_url}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT_SECS,
    )
    assert direct_resp.status_code == 200, direct_resp.text
    direct_choice = direct_resp.json()["choices"][0]
    assert "prompt_token_ids" in direct_choice, "Sanity check failed: direct request missing prompt_token_ids"

    # Router
    router_resp = http_requests.post(
        f"{sglang_router}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT_SECS,
    )
    assert router_resp.status_code == 200, router_resp.text
    router_choice = router_resp.json()["choices"][0]
    assert "prompt_token_ids" in router_choice, (
        "prompt_token_ids missing in router response — the sglang router dropped "
        "the return_prompt_token_ids field during request forwarding"
    )


@pytest.mark.system
def test_all_session_fields_passthrough(sglang_server, sglang_router):
    """All fields that the session server injects must survive the router
    simultaneously (this is the real-world scenario)."""
    extra = {
        "logprobs": True,
        "return_meta_info": True,
        "return_prompt_token_ids": True,
    }
    payload = _build_payload(extra)

    # Direct
    direct_resp = http_requests.post(
        f"{sglang_server.base_url}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT_SECS,
    )
    assert direct_resp.status_code == 200, direct_resp.text
    direct_data = direct_resp.json()
    direct_choice = direct_data["choices"][0]

    # Router
    router_resp = http_requests.post(
        f"{sglang_router}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT_SECS,
    )
    assert router_resp.status_code == 200, router_resp.text
    router_data = router_resp.json()
    router_choice = router_data["choices"][0]

    # Verify all three fields survived
    for field in ("meta_info", "prompt_token_ids"):
        assert field in router_choice, f"{field} missing in router response"

    # logprobs should be present
    assert router_choice.get("logprobs") is not None, "logprobs missing in router response"

    # Content should match between direct and router
    direct_text = direct_choice["message"]["content"]
    router_text = router_choice["message"]["content"]
    assert direct_text == router_text, f"response text mismatch:\n  direct: {direct_text!r}\n  router: {router_text!r}"

    # prompt_token_ids should match
    assert (
        direct_choice["prompt_token_ids"] == router_choice["prompt_token_ids"]
    ), "prompt_token_ids mismatch between direct and router responses"
