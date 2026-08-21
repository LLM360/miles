import asyncio
import threading

import numpy as np
import pytest
from tests.fast.router.test_session_samples_op import _ACCUMULATED, _build_core, _make_session, _two_turn_records
from tests.fast.router.test_session_worker_concurrency import _core

from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.utils.types import Sample


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("cancel", [False, True])
async def test_assembly_worker_preserves_lock_until_finished(monkeypatch, version, cancel):
    core, sid = _core(version)
    session = core.registry.get_session(sid)
    name = "_assemble_samples" if version == 1 else "_assemble_leaf_samples"
    original = getattr(core, name)
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()

    def assemble(*args):
        assert threading.get_ident() != main_thread
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(core, name, assemble)
    task = asyncio.create_task(core.collect_samples(sid, max_seq_len=None))
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        assert session.lock.locked()
        assert (await asyncio.wait_for(core.health(), 1)).status_code == 200
        if cancel:
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done() and session.lock.locked()
    finally:
        release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert (await task).status_code == 200
    assert not session.lock.locked()


@pytest.mark.parametrize("max_seq_len", [None, 8])
@pytest.mark.parametrize("early_length", [False, True])
async def test_token_only_assembly_preserves_training_fields_without_decoding(monkeypatch, max_seq_len, early_length):
    core = _build_core()
    records = _two_turn_records()
    if early_length:
        records[0].response["choices"][0]["finish_reason"] = "length"
    sid = await _make_session(core, records, _ACCUMULATED)
    # This fixture uses injected token IDs; mismatch checking is independent.
    core.registry.compute_session_mismatch = lambda session: None
    reference_response = await core.collect_samples(sid, max_seq_len=max_seq_len)
    reference = decode_samples_and_merge_input_sample(reference_response.body, Sample())

    def no_decode(*args, **kwargs):
        raise AssertionError("token-only sample assembly must not decode text")

    monkeypatch.setattr(core.registry.tokenizer, "decode", no_decode)
    response = await core.collect_samples(sid, max_seq_len=max_seq_len, decode_response=False)
    assert response.status_code == 200, response.body
    reply = decode_samples_and_merge_input_sample(response.body, Sample())
    sample, expected = reply.samples[0], reference.samples[0]
    for field in ("tokens", "response_length", "loss_mask", "rollout_log_probs", "status", "weight_versions"):
        assert getattr(sample, field) == getattr(expected, field), field
    assert sample.response == "" and sample.metadata["response_decoded"] is False
    assert "accumulated_token_ids" not in reply.session_metadata
    assert reply.session_metadata["accumulated_token_count"] == len(sample.tokens)
    assert reply.session_metadata["records_total"] == 2
    assert reply.session_metadata["records_dropped_after_first_non_completed"] == int(early_length)
    if sample.rollout_routed_experts is not None:
        np.testing.assert_array_equal(sample.rollout_routed_experts, expected.rollout_routed_experts)
