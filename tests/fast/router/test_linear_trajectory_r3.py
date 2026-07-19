import pybase64

from miles.rollout.session.linear_trajectory import LinearTrajectory
from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils.message_matcher_hub import strict_message_matches


def _record(routed_experts: str) -> SessionRecord:
    return SessionRecord(
        timestamp=0.0,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={},
        response={
            "choices": [
                {
                    "meta_info": {
                        "output_token_logprobs": [],
                        "routed_experts": routed_experts,
                    }
                }
            ]
        },
    )


def test_append_record_keeps_only_session_level_routed_experts():
    session = LinearTrajectory(trajectory_token_ids=[[1, 2, 3]])
    first = pybase64.b64encode(b"first-routing").decode("ascii")
    record = _record(first)

    session.append_record(record)

    assert session.latest_rollout_routed_experts == first
    assert session.latest_rollout_routed_experts_num_tokens == 3
    assert "routed_experts" not in record.response["choices"][0]["meta_info"]

    session.trajectory_token_ids.append([1, 2, 3, 4])
    second = pybase64.b64encode(b"second-routing").decode("ascii")
    second_record = _record(second)
    session.append_record(second_record)

    assert session.latest_rollout_routed_experts == second
    assert session.latest_rollout_routed_experts_num_tokens == 4
    assert all("routed_experts" not in stored.response["choices"][0]["meta_info"] for stored in session.records)


def test_routed_experts_are_truncated_to_rollback_checkpoint():
    # Four bytes per routed token row, three rows for a four-token checkpoint.
    rows = b"aaaabbbbcccc"
    session = LinearTrajectory(
        trajectory_token_ids=[[1, 2], [1, 2, 3, 4]],
        latest_rollout_routed_experts=pybase64.b64encode(rows).decode("ascii"),
        latest_rollout_routed_experts_num_tokens=4,
    )

    session._truncate_latest_rollout_routed_experts(2)

    assert session.latest_rollout_routed_experts_num_tokens == 2
    assert pybase64.b64decode(session.latest_rollout_routed_experts.encode("ascii")) == b"aaaa"


def test_reseed_clears_session_level_routed_experts():
    session = LinearTrajectory(
        messages=[{"role": "user", "content": "old"}],
        records=[_record("routing")],
        trajectory_token_ids=[[1, 2]],
        latest_rollout_routed_experts="routing",
        latest_rollout_routed_experts_num_tokens=2,
    )

    reseeded = session._try_detect_and_rollback_to_assistant_checkpoint(
        [{"role": "user", "content": "new"}], strict_message_matches
    )

    assert reseeded is None
    assert session.latest_rollout_routed_experts is None
    assert session.latest_rollout_routed_experts_num_tokens == 0


async def test_samples_wire_uses_latest_prefix_after_early_turn_selection():
    import numpy as np
    from tests.fast.router.test_session_samples_op import (
        _ACCUMULATED,
        _build_core,
        _expected_r3,
        _make_session,
        _two_turn_records,
    )

    from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
    from miles.utils.types import Sample

    core = _build_core()
    records = _two_turn_records()
    records[0].response["choices"][0]["finish_reason"] = "length"
    sid = await _make_session(core, records, _ACCUMULATED)
    session = core.registry.get_session(sid)
    assert all("routed_experts" not in record.response["choices"][0]["meta_info"] for record in session.records)
    response = await core.collect_samples(sid, max_seq_len=None)
    assert response.status_code == 200, response.body
    (sample,) = decode_samples_and_merge_input_sample(response.body, Sample()).samples
    np.testing.assert_array_equal(sample.rollout_routed_experts, _expected_r3(100, 8)[: len(sample.tokens) - 1])
    assert sample.metadata["dropped_trailing_turns"] == 1
