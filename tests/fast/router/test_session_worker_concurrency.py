"""Real session state remains consistent across worker awaits and cancellation."""

import asyncio
import itertools
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import requests
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.router.test_linear_trajectory import _make_registry
from tests.fast.router.test_sessions import router_env  # noqa: F401
from tests.fast.router.test_sessions_v2 import _serve_router
from tests.fast.router.test_stable_session_recovery import _Backend, _chat, _result

from miles.rollout.session import core as core_v1
from miles.rollout.session.concurrency import run_session_worker
from miles.rollout.session.linear_trajectory import LinearTrajectory
from miles.rollout.session.v2 import core as core_v2
from miles.rollout.session.v2.session_state import SessionRegistryV2


def _core(version):
    registry = _make_registry()
    cls = core_v1.SessionCore
    if version == 2:
        registry = SessionRegistryV2(None, tito_tokenizer=registry.tito_tokenizer)
        cls = core_v2.SessionCoreV2
    registry.compute_session_mismatch = lambda session: None
    config = make_session_server_config(
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    core = cls(_Backend(itertools.repeat(_result())), registry, config)
    sid = registry.create_session()
    return core, sid


def _target(version, phase):
    if version == 1:
        return LinearTrajectory, "prepare_pretokenized" if phase == "prepare" else "update_pretokenized_state"
    return core_v2, "prepare_pretokenized" if phase == "prepare" else "commit_generation"


async def _wait_for_worker(entered):
    async with asyncio.timeout(5):
        while not entered.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("phase", ["prepare", "commit"])
@pytest.mark.parametrize("cancel", [False, True])
def test_reads_and_cancellation_wait_for_complete_state(monkeypatch, version, phase, cancel):
    async def scenario():
        core, sid = _core(version)
        session = core.registry.get_session(sid)
        entered, release = threading.Event(), threading.Event()
        owner, method = _target(version, phase)
        original = getattr(owner, method)

        def blocked(state, *args, **kwargs):
            result = original(state, *args, **kwargs)
            if state is session:
                entered.set()
                assert release.wait(5), "worker was never released"
            return result

        # Exercise the real read lock and state capture without testing token
        # decoding against the unit registry's intentionally absent tokenizer.
        sample_states = []

        def assemble(*args, **kwargs):
            sample_states.append(json.loads(core._get_test_snapshot()))
            return []

        core._get_test_snapshot = lambda: json.dumps(
            {
                "records": len(session.records if version == 1 else session.active_records()),
                "tokens": session.token_ids if version == 1 else session.active_token_ids(),
            }
        )
        monkeypatch.setattr(
            core_v1 if version == 1 else core_v2,
            "compute_samples_from_openai_records" if version == 1 else "build_leaf_material",
            assemble,
        )
        monkeypatch.setattr(owner, method, blocked)
        chat = asyncio.create_task(_chat(core, sid))
        reads = []
        try:
            await _wait_for_worker(entered)
            assert session.lock.locked()
            assert (await asyncio.wait_for(core.health(), 1)).status_code == 200
            other = core.registry.create_session()
            assert (await asyncio.wait_for(_chat(core, other), 1)).status_code == 200
            reads = [
                asyncio.create_task(core.get_session(sid)),
                asyncio.create_task(core.collect_samples(sid, max_seq_len=None)),
            ]
            await asyncio.sleep(0.01)
            assert all(not task.done() for task in reads)
            if cancel:
                chat.cancel()
                await asyncio.sleep(0.01)
                chat.cancel()
                await asyncio.sleep(0.01)
                assert not chat.done() and session.lock.locked()
        finally:
            release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await chat
        else:
            assert (await chat).status_code == 200
        snapshot = json.loads((await reads[0]).body)
        assert (await reads[1]).status_code == 200
        expected_turns = 0 if phase == "prepare" else 1
        assert len(snapshot["records"]) == expected_turns
        assert snapshot["metadata"]["accumulated_token_ids"] == ([0, 10] if expected_turns else [])
        assert all(state == {"records": 1, "tokens": [0, 10]} for state in sample_states)
        final = json.loads((await core.get_session(sid)).body)
        assert len(final["records"]) == (0 if cancel and phase == "prepare" else 1)
        assert not session.lock.locked()

    asyncio.run(scenario())


@pytest.mark.parametrize("version", [1, 2])
def test_same_session_preparation_is_serialized(monkeypatch, version):
    async def scenario():
        core, sid = _core(version)
        owner, method = _target(version, "prepare")
        original = getattr(owner, method)
        entered, release = threading.Event(), threading.Event()
        calls = []

        def blocked(*args, **kwargs):
            calls.append(threading.get_ident())
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, method, blocked)
        first = asyncio.create_task(_chat(core, sid))
        await _wait_for_worker(entered)
        second = asyncio.create_task(_chat(core, sid))
        try:
            await asyncio.sleep(0.02)
            assert len(calls) == 1
        finally:
            release.set()
        assert all(response.status_code == 200 for response in await asyncio.gather(first, second))
        assert len(calls) == 2

    asyncio.run(scenario())


def test_worker_exception_propagates_and_releases_lock():
    async def scenario():
        lock = asyncio.Lock()

        def fail():
            raise ValueError("tokenization failure")

        with pytest.raises(ValueError, match="tokenization failure"):
            async with lock:
                await run_session_worker(fail)
        assert not lock.locked()

    asyncio.run(scenario())


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("phase", ["prepare", "commit"])
def test_http_health_progresses_and_chat_succeeds(request, version, phase):
    def check(env):
        sid = requests.post(f"{env.url}/sessions", timeout=5).json()["session_id"]
        entered, release = threading.Event(), threading.Event()
        owner, method = _target(version, phase)
        original = getattr(owner, method)

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)

        with patch.object(owner, method, blocked), ThreadPoolExecutor(max_workers=1) as pool:
            chat = pool.submit(
                requests.post,
                f"{env.url}/sessions/{sid}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
                timeout=10,
            )
            try:
                assert entered.wait(5), "chat never reached worker"
                assert requests.get(f"{env.url}/health", timeout=1).status_code == 200
            finally:
                release.set()
            assert chat.result().status_code == 200

    if version == 1:
        check(request.getfixturevalue("router_env"))
    else:
        with _serve_router() as env:
            check(env)
