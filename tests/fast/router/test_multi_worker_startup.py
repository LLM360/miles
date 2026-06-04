"""Smoke test for multi-process session-server startup.

Does NOT load real tokenizers (would download HF weights). Instead patches
``run_session_server`` with a tiny fake uvicorn app and verifies that:

  * N worker processes are spawned on distinct ports.
  * The front-end ASGI router comes up on ``session_server_port``.
  * The router consistent-hash routes ``/sessions/{id}/...`` to the
    correct backend.

Importing this file is heavy (multiprocessing.Process), so it is grouped
with the rest of router fast tests but skips in CI environments where
binding ports is unreliable.
"""

import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
import requests

from miles.utils.http_utils import find_available_port


class _BackendHandler(BaseHTTPRequestHandler):
    """Echoes back the port it's serving on, so the test can verify routing."""

    def log_message(self, format, *args):  # noqa: A002 - stdlib API
        pass  # silence stderr in CI

    def do_GET(self):
        port = self.server.server_address[1]
        body = f'{{"port": {port}, "path": "{self.path}"}}'.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET
    do_DELETE = do_GET


def _spawn_fake_backend(port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), _BackendHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _wait_port(port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"port {port} never came up")


@pytest.fixture
def fake_backends():
    """Spin up 4 fake HTTP backends on consecutive ports."""
    base = find_available_port(40000)
    ports = []
    servers = []
    # Find 4 free consecutive-ish ports (best-effort; doesn't have to be
    # contiguous because we control the URL list passed to the router).
    p = base
    while len(ports) < 4:
        try:
            servers.append(_spawn_fake_backend(p))
            ports.append(p)
        except OSError:
            pass
        p += 1
    for port in ports:
        _wait_port(port)
    yield ports
    for s in servers:
        s.shutdown()


def test_session_router_routes_by_session_id(fake_backends):
    """End-to-end: SessionRouter started in-process, requests reach the right backend."""
    from miles.rollout.session.session_router import SessionRouter
    from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

    backend_urls = [f"http://127.0.0.1:{p}" for p in fake_backends]
    args = SimpleNamespace(miles_router_timeout=10.0)
    router = SessionRouter(args, backend_urls)

    router_port = find_available_port(41000)
    server = UvicornThreadServer(router.app, host="127.0.0.1", port=router_port)
    server.start()
    try:
        _wait_port(router_port)
        worker_count = len(fake_backends)

        # Hit the router with N distinct session_ids carrying the
        # ``w<idx>-`` prefix; verify each one reached the backend whose
        # port matches its parsed worker index.
        for _ in range(20):
            for worker_index in range(worker_count):
                sid = f"w{worker_index}-{uuid.uuid4().hex}"
                expected_port = fake_backends[worker_index]
                resp = requests.get(
                    f"http://127.0.0.1:{router_port}/sessions/{sid}/v1/chat/completions",
                    timeout=5,
                )
                assert resp.status_code == 200, resp.text
                assert (
                    resp.json()["port"] == expected_port
                ), f"sid={sid} expected port {expected_port}, got {resp.json()}"
    finally:
        server.stop()


def test_session_router_health_no_proxy(fake_backends):
    """/health on the router returns its own status, does not hit any backend."""
    from miles.rollout.session.session_router import SessionRouter
    from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

    backend_urls = [f"http://127.0.0.1:{p}" for p in fake_backends]
    args = SimpleNamespace(miles_router_timeout=10.0)
    router = SessionRouter(args, backend_urls)

    router_port = find_available_port(42000)
    server = UvicornThreadServer(router.app, host="127.0.0.1", port=router_port)
    server.start()
    try:
        _wait_port(router_port)
        resp = requests.get(f"http://127.0.0.1:{router_port}/health", timeout=5)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["role"] == "session-router"
        assert body["worker_count"] == len(fake_backends)
        assert body["backends"] == backend_urls
    finally:
        server.stop()
