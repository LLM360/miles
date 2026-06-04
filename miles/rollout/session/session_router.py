"""ASGI front-end for the multi-process session server.

When ``--session-server-workers N`` is set with N > 1, ``_start_session_server``
spawns N backend SessionServer processes on consecutive ports and runs this
front-end on ``args.session_server_port``. The front-end:

  * Parses ``session_id`` from the URL path (``/sessions/{id}/...``).
  * Picks a backend via ``session_id_bucket(session_id, N)`` — the same
    hash used by ``SessionRegistry.create_session`` so a session always
    routes back to the worker that created it.
  * For the stateless ``POST /sessions`` and ``GET /health`` paths,
    routes by a round-robin counter (any worker will do; ``create_session``
    on the chosen worker guarantees the returned id hashes to itself).
  * Streams the response body through verbatim (no JSON re-encoding).

The router does almost no per-request CPU work (path-parse + hash +
httpx passthrough), so its GIL does not become the new bottleneck —
all the tokenizer / TITO work happens in the backend workers, each in
its own process.
"""

import itertools
import json
import logging
import re

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from miles.rollout.session.linear_trajectory import session_id_bucket

logger = logging.getLogger(__name__)

# Matches /sessions/{id}/... and captures {id}. Bare POST /sessions
# (creating a new session, no id yet) is intentionally excluded.
_SESSION_PATH_RE = re.compile(r"^/sessions/([^/]+)(?:/|$)")


class SessionRouter:
    """FastAPI app that hash-routes session requests to backend workers."""

    def __init__(self, args, backend_urls: list[str]):
        if not backend_urls:
            raise ValueError("SessionRouter requires at least one backend URL")
        self.backend_urls = backend_urls
        self.worker_count = len(backend_urls)
        self.app = FastAPI()

        timeout = getattr(args, "miles_router_timeout", 600.0)
        # max_keepalive_connections=0 mirrors init_http_client: prevents
        # all router->backend traffic from pinning to one TCP connection
        # against one backend worker.
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=0),
            timeout=httpx.Timeout(timeout),
        )
        self.app.router.on_shutdown.append(self.client.aclose)

        # Round-robin counter for stateless paths (e.g. POST /sessions).
        self._rr_counter = itertools.count()

        self._setup_routes()

    def pick_backend(self, path: str) -> str:
        """Pick a backend URL for ``path``.

        Stateful paths (``/sessions/{id}/...``) hash by session_id.
        Stateless paths round-robin so we don't hot-spot worker 0.
        """
        m = _SESSION_PATH_RE.match(path)
        if m is not None:
            session_id = m.group(1)
            idx = session_id_bucket(session_id, self.worker_count)
        else:
            idx = next(self._rr_counter) % self.worker_count
        return self.backend_urls[idx]

    async def proxy(self, request: Request) -> Response:
        path = request.url.path
        backend = self.pick_backend(path)
        url = f"{backend}{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        body = await request.body()
        # Strip framing / host headers — httpx will recompute them and
        # we already mirror what session_server.py does on its own proxy
        # path.
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "host")
        }

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
        except httpx.TransportError as exc:
            logger.warning(
                "[session-router] backend transport error: %s %s -> %s: %s",
                request.method, path, backend, exc,
            )
            return JSONResponse(
                status_code=502,
                content={"error": f"session-router backend transport error: {type(exc).__name__}: {exc}"},
            )

        content = await response.aread()
        resp_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "server")
        }
        # Try to mirror the backend's content shape. JSONResponse re-encodes
        # which guarantees a correct content-length even if our header
        # stripping changed the wire shape; fall back to raw bytes when the
        # body is not JSON.
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=response.status_code, headers=resp_headers)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Response(
                content=content,
                status_code=response.status_code,
                headers=resp_headers,
                media_type=resp_headers.get("content-type", ""),
            )

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        async def health():
            # Front-end-local health: do NOT proxy. Operators want to
            # tell "is the router itself up?" separately from "is any
            # backend up?". Per-worker health is reachable via the
            # backend ports directly during debugging.
            return {
                "status": "ok",
                "role": "session-router",
                "worker_count": self.worker_count,
                "backends": self.backend_urls,
            }

        @self.app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )
        async def catchall(request: Request, path: str):
            return await self.proxy(request)


def run_session_router(args, backend_urls: list[str]):
    """Entry point for the front-end process started by _start_session_server."""
    setproctitle.setproctitle("miles-session-router")
    router = SessionRouter(args, backend_urls)
    logger.info(
        "[session-router] Starting on %s:%s, routing to %d backends: %s",
        args.session_server_ip,
        args.session_server_port,
        len(backend_urls),
        backend_urls,
    )
    uvicorn.run(
        router.app,
        host=args.session_server_ip,
        port=args.session_server_port,
        log_level="info",
    )
