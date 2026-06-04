"""Unit tests for multi-process session routing.

These verify the load-bearing invariant of the multi-process session-server
design: every session_id that ``SessionRegistry.create_session`` returns
parses — via the prefix-encoding contract — back to the worker that
created it. If this ever breaks, sticky routing breaks, and the
session-server falls back to the auto-reseed path on every turn (silently
losing state).
"""

from types import SimpleNamespace
from typing import Any

import pytest

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.session_router import parse_worker_index
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer


class _MockTITOTokenizer(TITOTokenizer):
    """Stub: no real tokenizer work needed for routing tests."""

    def create_comparator(self):
        return None

    def tokenize_additional_non_assistant(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return []

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return list(pretokenized_token_ids)


def _make_registry(worker_index: int, worker_count: int) -> SessionRegistry:
    args = SimpleNamespace()
    mock_tito = _MockTITOTokenizer(
        tokenizer=None,
        assistant_start_str="<|im_start|>assistant",
        allowed_append_roles=None,
    )
    return SessionRegistry(
        args,
        tokenizer=None,
        tito_tokenizer=mock_tito,
        worker_index=worker_index,
        worker_count=worker_count,
    )


class TestParseWorkerIndex:
    def test_parses_well_formed_prefix(self):
        assert parse_worker_index("w0-abcdef", 4) == 0
        assert parse_worker_index("w3-abcdef", 4) == 3
        assert parse_worker_index("w7-deadbeef", 8) == 7

    def test_rejects_missing_prefix(self):
        # Bare uuid hex (the single-worker shape) has no w<idx>- prefix.
        with pytest.raises(ValueError):
            parse_worker_index("a" * 32, 4)

    def test_rejects_non_numeric_prefix(self):
        with pytest.raises(ValueError):
            parse_worker_index("wx-abcdef", 4)

    def test_rejects_out_of_range_index(self):
        # Worker minted with worker_count=8 but router is now running
        # with worker_count=4 — the parsed index is out of range.
        with pytest.raises(ValueError):
            parse_worker_index("w5-abcdef", 4)

    def test_rejects_negative_index_via_no_match(self):
        # Regex `^w(\d+)-` doesn't accept a minus sign, so this is a
        # missing-prefix failure rather than an out-of-range failure.
        with pytest.raises(ValueError):
            parse_worker_index("w-1-abcdef", 4)


class TestSessionRegistryRouting:
    @pytest.mark.parametrize("worker_count", [1, 2, 4, 8, 16])
    def test_create_session_id_parses_to_self(self, worker_count: int):
        """Every session_id created by worker i must parse back to index i."""
        for worker_index in range(worker_count):
            registry = _make_registry(worker_index, worker_count)
            for _ in range(50):
                sid = registry.create_session()
                if worker_count == 1:
                    # Single-worker deployments keep emitting bare uuid hex
                    # for back-compat; routing is trivially worker 0.
                    assert len(sid) == 32
                else:
                    assert parse_worker_index(sid, worker_count) == worker_index, (
                        f"session_id {sid} from worker {worker_index}/{worker_count} "
                        f"did not parse to its own index"
                    )

    def test_default_single_worker_behavior(self):
        """worker_count=1 is the existing behavior; bare uuid hex."""
        registry = _make_registry(0, 1)
        sid = registry.create_session()
        assert len(sid) == 32  # uuid4 hex, no prefix
        assert sid in registry.sessions

    def test_multi_worker_ids_carry_prefix(self):
        """worker_count>1 ids must carry the Stripe-style w<idx>- prefix."""
        registry = _make_registry(worker_index=2, worker_count=8)
        sid = registry.create_session()
        assert sid.startswith("w2-")
        assert sid in registry.sessions

    def test_invalid_worker_index(self):
        with pytest.raises(ValueError):
            _make_registry(worker_index=5, worker_count=4)
        with pytest.raises(ValueError):
            _make_registry(worker_index=-1, worker_count=4)

    def test_invalid_worker_count(self):
        with pytest.raises(ValueError):
            _make_registry(worker_index=0, worker_count=0)


class TestRouterAgreement:
    """The front-end router and SessionRegistry must agree on the routing
    contract — now a prefix parse, not a hash. With prefix encoding there
    is no "shared algorithm" to drift on, but the contract still has to
    hold end-to-end.
    """

    @pytest.mark.parametrize("worker_count", [2, 3, 4, 7, 8])
    def test_session_router_pick_matches_creator(self, worker_count: int):
        from miles.rollout.session.session_router import SessionRouter

        args = SimpleNamespace(miles_router_timeout=1.0)
        backends = [f"http://127.0.0.1:{6000 + i}" for i in range(worker_count)]
        router = SessionRouter(args, backends)

        for worker_index in range(worker_count):
            registry = _make_registry(worker_index, worker_count)
            for _ in range(20):
                sid = registry.create_session()
                # The router's pick_backend on a stateful path must
                # match the URL of the worker that created the session.
                picked = router.pick_backend(f"/sessions/{sid}/v1/chat/completions")
                assert picked == backends[worker_index], (
                    f"session_id={sid} created by worker {worker_index} " f"but router routed to {picked}"
                )

    def test_router_unknown_session_id_falls_back_to_round_robin(self):
        """Malformed/out-of-range session_ids round-robin to a backend.

        Rolling-deploy safety net: rather than 404, the router falls
        back to round-robin so the backend's ``get_or_create_session``
        can reseed. See PR #31 finding M.
        """
        from miles.rollout.session.session_router import SessionRouter

        args = SimpleNamespace(miles_router_timeout=1.0)
        backends = [f"http://127.0.0.1:{6000 + i}" for i in range(4)]
        router = SessionRouter(args, backends)

        # No w<idx>- prefix -> round-robin, never raises.
        picks_no_prefix = [router.pick_backend("/sessions/badid_no_prefix/v1/chat/completions") for _ in range(40)]
        assert set(picks_no_prefix) == set(backends)

        # Out-of-range worker index (id minted under wider fleet) -> round-robin.
        picks_oor = [router.pick_backend("/sessions/w9-deadbeef/v1/chat/completions") for _ in range(40)]
        assert set(picks_oor) == set(backends)

    def test_router_stateless_paths_round_robin(self):
        """POST /sessions (no id) and other unmatched paths should not pin to one backend."""
        from miles.rollout.session.session_router import SessionRouter

        args = SimpleNamespace(miles_router_timeout=1.0)
        backends = [f"http://127.0.0.1:{7000 + i}" for i in range(4)]
        router = SessionRouter(args, backends)
        picks = [router.pick_backend("/sessions") for _ in range(40)]
        # Round-robin: every backend must appear at least once.
        assert set(picks) == set(backends)
