"""Unit tests for the multi-worker SO_REUSEPORT path in ``run_session_router``.

The router is stateless (per-process round-robin counter, pure-function
URL-prefix routing), so running K parallel uvicorn workers behind the
same port is safe. These tests pin down the launch-time contract:

  * K=1 (default): existing single-process behavior is preserved exactly
    (``uvicorn.run`` is called with the same kwargs as before; no
    SO_REUSEPORT socket is opened).
  * K>1: a SO_REUSEPORT socket is opened and handed to
    ``uvicorn.Server.run(sockets=[sock])``.
  * Per-router-worker args copy carries a distinguishable
    ``session_server_instance_id`` so log lines from different worker
    PIDs can be told apart (``-router{i}`` suffix applied in
    ``_start_session_server``).

These tests do NOT spawn real processes or bind real ports — they patch
the uvicorn entrypoints and assert on the call arguments. That makes
them fast and safe to run on macOS CI (SO_REUSEPORT semantics differ
there; the multi-worker path is Linux-only at runtime).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# run_session_router uvicorn invocation
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_router_args():
    """Minimal args namespace accepted by SessionRouter + run_session_router."""
    return SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_port=0,  # we never actually bind in these tests
        session_server_instance_id="test-instance",
        miles_router_timeout=10.0,
        session_router_workers=1,
    )


def test_run_session_router_single_worker_calls_uvicorn_run(fake_router_args):
    """K=1 path: must call ``uvicorn.run`` with the legacy kwargs, no
    sockets, no SO_REUSEPORT.

    This is the backward-compat guarantee — production deployments that
    don't opt in must see literally zero behavior change.
    """
    from miles.rollout.session import session_router as sr

    backend_urls = ["http://127.0.0.1:6001", "http://127.0.0.1:6002"]
    with patch.object(sr, "uvicorn") as mock_uvicorn, patch.object(sr, "_make_reuseport_socket") as mock_sock:
        sr.run_session_router(fake_router_args, backend_urls)
        # The legacy code-path is uvicorn.run(...). Server / Config must
        # not be touched, and crucially no SO_REUSEPORT socket is opened.
        mock_uvicorn.run.assert_called_once()
        _args, kwargs = mock_uvicorn.run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 0
        assert kwargs["log_level"] == "info"
        mock_uvicorn.Config.assert_not_called()
        mock_uvicorn.Server.assert_not_called()
        mock_sock.assert_not_called()


def test_run_session_router_multi_worker_uses_reuseport_socket(fake_router_args):
    """K>1 path: must open a SO_REUSEPORT socket and hand it to
    ``uvicorn.Server.run(sockets=[sock])``. ``uvicorn.run`` must NOT
    be called (it would bind its own socket and conflict).
    """
    from miles.rollout.session import session_router as sr

    fake_router_args.session_router_workers = 4
    backend_urls = ["http://127.0.0.1:6001", "http://127.0.0.1:6002"]
    fake_sock = object()
    with patch.object(sr, "uvicorn") as mock_uvicorn, patch.object(
        sr, "_make_reuseport_socket", return_value=fake_sock
    ) as mock_sock:
        sr.run_session_router(fake_router_args, backend_urls)
        # Legacy uvicorn.run MUST NOT be used in the multi-worker path.
        mock_uvicorn.run.assert_not_called()
        # SO_REUSEPORT socket must be opened at the configured host:port.
        mock_sock.assert_called_once_with("127.0.0.1", 0)
        # Server.run must receive the pre-bound socket.
        mock_uvicorn.Config.assert_called_once()
        cfg_args, cfg_kwargs = mock_uvicorn.Config.call_args
        assert cfg_kwargs["host"] == "127.0.0.1"
        assert cfg_kwargs["port"] == 0
        assert cfg_kwargs["log_level"] == "info"
        # Server() instantiated with the Config; .run(sockets=[sock]) called.
        mock_uvicorn.Server.assert_called_once()
        server_instance = mock_uvicorn.Server.return_value
        server_instance.run.assert_called_once_with(sockets=[fake_sock])


def test_run_session_router_default_treats_missing_attr_as_one(fake_router_args):
    """A pre-existing args object built before this PR landed will not have
    ``session_router_workers``. Treat that the same as K=1 (legacy path)
    so a partial deploy doesn't silently flip behavior.
    """
    from miles.rollout.session import session_router as sr

    del fake_router_args.session_router_workers
    backend_urls = ["http://127.0.0.1:6001"]
    with patch.object(sr, "uvicorn") as mock_uvicorn, patch.object(sr, "_make_reuseport_socket") as mock_sock:
        sr.run_session_router(fake_router_args, backend_urls)
        mock_uvicorn.run.assert_called_once()
        mock_sock.assert_not_called()


# ---------------------------------------------------------------------------
# per-worker args copy: -router{i} suffix on session_server_instance_id
# ---------------------------------------------------------------------------


def test_per_worker_args_copy_isolated_from_caller():
    """``_per_worker_args_copy`` must return an independent copy so we can
    safely overwrite ``session_server_instance_id`` per worker without
    aliasing the caller's args.

    The router-worker spawn loop in ``_start_session_server`` stamps
    ``-router{i}`` onto the copy; this test pins down that the stamp on
    one copy does not bleed into the next copy or the original.
    """
    from miles.ray.rollout import _per_worker_args_copy

    args = SimpleNamespace(session_server_instance_id="base-id", some_list=[1, 2, 3])
    c0 = _per_worker_args_copy(args)
    c1 = _per_worker_args_copy(args)
    c0.session_server_instance_id = f"{args.session_server_instance_id}-router0"
    c1.session_server_instance_id = f"{args.session_server_instance_id}-router1"
    assert c0.session_server_instance_id == "base-id-router0"
    assert c1.session_server_instance_id == "base-id-router1"
    # Caller's args is untouched.
    assert args.session_server_instance_id == "base-id"
    # And the nested mutable is deep-copied — mutating c0 doesn't leak
    # into c1 or the original (this is the invariant `_per_worker_args_copy`
    # exists to guarantee).
    c0.some_list.append(99)
    assert args.some_list == [1, 2, 3]
    assert c1.some_list == [1, 2, 3]
