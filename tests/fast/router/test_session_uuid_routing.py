"""Unit tests for multi-process session routing.

These verify the load-bearing invariant of the multi-process session-server
design: every session_id that ``SessionRegistry.create_session`` returns
hashes — via the same function the front-end router uses — to the worker
that created it. If this ever breaks, sticky routing breaks, and the
session-server falls back to the auto-reseed path on every turn (silently
losing state).
"""

from types import SimpleNamespace
from typing import Any

import pytest

from miles.rollout.session.linear_trajectory import SessionRegistry, session_id_bucket
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


class TestSessionIdBucket:
    def test_single_worker_always_bucket_zero(self):
        for sid in ("a" * 32, "deadbeef" * 4, "0" * 32):
            assert session_id_bucket(sid, 1) == 0

    def test_deterministic(self):
        sid = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        assert session_id_bucket(sid, 4) == session_id_bucket(sid, 4)
        assert session_id_bucket(sid, 8) == session_id_bucket(sid, 8)

    def test_distribution_is_reasonable(self):
        """With many uuids, buckets should be roughly balanced (sanity, not strict)."""
        import uuid

        counts = [0] * 8
        for _ in range(8000):
            counts[session_id_bucket(uuid.uuid4().hex, 8)] += 1
        # Expect ~1000 per bucket; allow wide margin to avoid flakes.
        assert all(500 <= c <= 1500 for c in counts), counts


class TestSessionRegistryRouting:
    @pytest.mark.parametrize("worker_count", [1, 2, 4, 8, 16])
    def test_create_session_id_hashes_to_self(self, worker_count: int):
        """Every session_id created by worker i must hash back to bucket i."""
        for worker_index in range(worker_count):
            registry = _make_registry(worker_index, worker_count)
            for _ in range(50):
                sid = registry.create_session()
                assert session_id_bucket(sid, worker_count) == worker_index, (
                    f"session_id {sid} from worker {worker_index}/{worker_count} "
                    f"hashed to bucket {session_id_bucket(sid, worker_count)}"
                )

    def test_default_single_worker_behavior(self):
        """worker_count=1 is the existing behavior; no regen needed."""
        registry = _make_registry(0, 1)
        sid = registry.create_session()
        assert len(sid) == 32  # uuid4 hex
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
    """The front-end router and SessionRegistry must agree on the hash.

    If anyone ever changes one without the other, these tests fail.
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
                    f"session_id={sid} created by worker {worker_index} "
                    f"but router routed to {picked}"
                )

    def test_router_stateless_paths_round_robin(self):
        """POST /sessions (no id) and other unmatched paths should not pin to one backend."""
        from miles.rollout.session.session_router import SessionRouter

        args = SimpleNamespace(miles_router_timeout=1.0)
        backends = [f"http://127.0.0.1:{7000 + i}" for i in range(4)]
        router = SessionRouter(args, backends)
        picks = [router.pick_backend("/sessions") for _ in range(40)]
        # Round-robin: every backend must appear at least once.
        assert set(picks) == set(backends)
