"""ASGI front-end for the multi-process session server.

When ``--session-server-workers N`` is set with N > 1, ``_start_session_server``
spawns N backend SessionServer processes on consecutive ports and runs this
front-end on ``args.session_server_port``. The front-end:

  * Parses ``session_id`` from the URL path (``/sessions/{id}/...``).
  * Routes by parsing the ``w<idx>-`` prefix stamped onto the id by
    ``SessionRegistry.create_session``. Prefix-encoded ids (Stripe-style)
    eliminate the hash-agreement risk between router and backend — there
    is no shared algorithm to drift on.
  * For the stateless ``POST /sessions`` and ``GET /health`` paths,
    routes by a round-robin counter (any worker will do; the chosen
    worker stamps its own index on the returned id).
  * Streams the response body through verbatim (no JSON re-encoding,
    no full-body buffering).

The router does almost no per-request CPU work (path-parse + str.split +
httpx passthrough), so its GIL does not become the new bottleneck —
all the tokenizer / TITO work happens in the backend workers, each in
its own process.
"""

import itertools
import logging
import re
import socket

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)

# Matches /sessions/{id}/... and captures {id}. Bare POST /sessions
# (creating a new session, no id yet) is intentionally excluded.
_SESSION_PATH_RE = re.compile(r"^/sessions/([^/]+)(?:/|$)")

# Matches the ``w<idx>-`` prefix that ``SessionRegistry.create_session``
# stamps onto every multi-worker session id.
_WORKER_PREFIX_RE = re.compile(r"^w(\d+)-")


def parse_worker_index(session_id: str, worker_count: int) -> int:
    """Parse the ``w<idx>-`` prefix and return the worker index.

    Raises ``ValueError`` if the prefix is missing or the parsed index
    is out of range for the current worker_count.
    """
    m = _WORKER_PREFIX_RE.match(session_id)
    if m is None:
        raise ValueError(f"session_id {session_id!r} does not have the expected 'w<idx>-' prefix")
    idx = int(m.group(1))
    if not 0 <= idx < worker_count:
        raise ValueError(
            f"session_id {session_id!r} parses to worker index {idx}, " f"out of range for worker_count={worker_count}"
        )
    return idx


class SessionRouter:
    """FastAPI app that hash-routes session requests to backend workers."""

    def __init__(self, args, backend_urls: list[str]):
        if not backend_urls:
            raise ValueError("SessionRouter requires at least one backend URL")
        self.backend_urls = backend_urls
        self.worker_count = len(backend_urls)
        # Cluster-facing instance id. ``OpenAIEndpointTracer.create``
        # (miles/rollout/generate_utils/openai_endpoint_utils.py) reads
        # this from ``/health`` and stamps it on trial metadata; if it
        # falls through to None, downstream assertions in the test suite
        # (test_sessions.py, test_multi_turn.py) regress. See PR #31 H1.
        self.session_server_instance_id = getattr(args, "session_server_instance_id", None)
        self.app = FastAPI()

        timeout = getattr(args, "miles_router_timeout", 600.0)
        # Connection pool sized for ~N backends * ~hundreds of in-flight
        # requests each. Keepalive MUST be enabled here: every request is
        # already explicitly routed by ``session_id``, so there's no
        # "pinning to one backend" risk (the analogy to
        # ``init_http_client``'s keepalive=0 setting does not apply).
        # With keepalive=0 every request did a full TCP handshake and
        # then sat in TIME_WAIT, leading to ephemeral port exhaustion
        # under sustained load (see PR #31 deep review B1).
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=4096, max_keepalive_connections=1024),
            timeout=httpx.Timeout(timeout),
        )
        self.app.router.on_shutdown.append(self.client.aclose)

        # Round-robin counter for stateless paths (e.g. POST /sessions).
        self._rr_counter = itertools.count()

        self._setup_routes()

    def pick_backend(self, path: str) -> str:
        """Pick a backend URL for ``path``.

        Stateful paths (``/sessions/{id}/...``) parse the ``w<idx>-``
        prefix stamped onto the id by ``SessionRegistry.create_session``
        and route to the owning backend. Stateless paths round-robin so
        we don't hot-spot worker 0.

        Rolling-deploy shrink safety net: if a session_id doesn't carry
        the ``w<idx>-`` prefix or names a worker outside
        ``[0, worker_count)`` (e.g. a trial in-flight from a previous
        deploy with a larger worker_count, or a legacy bare-uuid id from
        an N=1 run), we fall back to round-robin instead of 404.
        The chosen backend's ``get_or_create_session`` will reseed the
        session under a freshly-minted prefix; the trial loses state but
        recovers in-place rather than dying mid-rollout. See PR #31 M.
        """
        m = _SESSION_PATH_RE.match(path)
        if m is not None:
            session_id = m.group(1)
            try:
                idx = parse_worker_index(session_id, self.worker_count)
            except ValueError as exc:
                logger.info(
                    "[session-router] session_id %r doesn't route cleanly (%s); "
                    "falling back to round-robin for get_or_create reseed",
                    session_id,
                    exc,
                )
                # next() on itertools.count() is atomic under the CPython GIL.
                idx = next(self._rr_counter) % self.worker_count
        else:
            idx = next(self._rr_counter) % self.worker_count
        return self.backend_urls[idx]

    # Hop-by-hop headers per RFC 7230 §6.1 — these must NOT be forwarded
    # because they describe the single hop, not the end-to-end message.
    # httpx will recompute content-length / transfer-encoding itself.
    _HOP_BY_HOP_HEADERS = frozenset(
        {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
            "content-length",
        }
    )

    async def proxy(self, request: Request) -> Response:
        path = request.url.path
        # pick_backend no longer raises for malformed session_ids — it
        # falls back to round-robin so the backend's get_or_create_session
        # can reseed. See PR #31 M.
        backend = self.pick_backend(path)
        url = f"{backend}{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        # Strip framing / host headers — httpx will recompute them.
        # Stream the request body straight through (no buffering in
        # router RAM) so multi-MB SGLang request payloads don't OOM
        # the router at high concurrency.
        req_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in self._HOP_BY_HOP_HEADERS and k.lower() != "host"
        }

        # Use the streaming client.send path so both request and
        # response bodies are streamed end-to-end. We must close the
        # upstream response when the client disconnects; StreamingResponse
        # does that by raising inside the generator.
        upstream_req = self.client.build_request(
            request.method,
            url,
            content=request.stream(),
            headers=req_headers,
        )
        try:
            upstream_resp = await self.client.send(upstream_req, stream=True)
        except httpx.TransportError as exc:
            logger.warning(
                "[session-router] backend transport error: %s %s -> %s: %s",
                request.method,
                path,
                backend,
                exc,
            )
            # Log the full exception above; return a sanitized message to
            # the caller so internal backend hostnames / paths can't leak.
            return JSONResponse(
                status_code=502,
                content={"error": "session-router backend transport error"},
            )

        # Filter hop-by-hop response headers per RFC 7230 §6.1. Also
        # drop "server" (cosmetic — was stripped in the old buffered
        # path). Keep content-type as-is so charset hints survive.
        resp_headers = {
            k: v
            for k, v in upstream_resp.headers.items()
            if k.lower() not in self._HOP_BY_HOP_HEADERS and k.lower() != "server"
        }

        async def _body_stream():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    yield chunk
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(
            _body_stream(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=upstream_resp.headers.get("content-type"),
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
                # H1: must be present so OpenAIEndpointTracer.create can
                # stamp ``session_server_instance_id`` on trial metadata
                # in multi-worker mode (the router is the user-facing
                # ``session_url`` then).
                "session_server_instance_id": self.session_server_instance_id,
            }

        @self.app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )
        async def catchall(request: Request, path: str):
            return await self.proxy(request)


def _make_reuseport_socket(host: str, port: int) -> socket.socket:
    """Open a TCP listener bound with SO_REUSEPORT for kernel-level load balancing.

    Multiple processes can bind the same (host, port) when each one sets
    SO_REUSEPORT before bind(). The Linux kernel (>= 3.9) then hash-distributes
    incoming connections across the listener sockets, giving us K-way
    parallelism without a userspace dispatcher.

    Requires Linux. macOS has SO_REUSEPORT but with different semantics
    (last-bound wins for unicast); the multi-worker path is intended for
    Slurm/Linux production. The single-worker (K=1) path is unchanged.
    """
    family = socket.AF_INET6 if host and ":" in host else socket.AF_INET
    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    # Match uvicorn's own default backlog.
    sock.listen(2048)
    sock.set_inheritable(True)
    return sock


def run_session_router(args, backend_urls: list[str]):
    """Entry point for the front-end process started by _start_session_server.

    Honors ``args.session_router_workers`` (default 1):

      * K=1: identical to pre-existing behavior (``uvicorn.run`` binds the
        listening socket internally).
      * K>1: the caller has already forked K worker processes; this one
        opens a SO_REUSEPORT socket and hands it to
        ``uvicorn.Server.run(sockets=[sock])``. The Linux kernel
        distributes connections across the K workers.

    NOTE on the uvicorn API: as of uvicorn 0.40--0.49 (the versions we
    currently ship), ``uvicorn.Config`` does **not** have a ``reuse_port``
    kwarg / attribute. The supported path for pre-bound listeners is
    ``Server.run(sockets=[...])``. So we open the SO_REUSEPORT socket
    ourselves in each worker and pass it in. See PR description for the
    bench evidence motivating this change.
    """
    setproctitle.setproctitle("miles-session-router")
    router = SessionRouter(args, backend_urls)
    router_worker_count = max(1, int(getattr(args, "session_router_workers", 1) or 1))
    if router_worker_count <= 1:
        # Preserve exact pre-existing behavior — bit-for-bit identical to
        # the previous implementation, log line included.
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
        return

    logger.info(
        "[session-router] Starting on %s:%s, routing to %d backends: %s (router_workers=%d, SO_REUSEPORT)",
        args.session_server_ip,
        args.session_server_port,
        len(backend_urls),
        backend_urls,
        router_worker_count,
    )

    # K>1: open our own SO_REUSEPORT socket and hand it to uvicorn.Server.
    sock = _make_reuseport_socket(args.session_server_ip, args.session_server_port)
    config = uvicorn.Config(
        router.app,
        host=args.session_server_ip,
        port=args.session_server_port,
        log_level="info",
        loop="asyncio",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run(sockets=[sock])
