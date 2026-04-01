"""Tests for event_analyzer rules/weight_checksum."""

from miles.utils.event_analyzer.rules.weight_checksum import (
    ChecksumMismatchIssue,
    _flatten_event,
    _flatten_nested,
    check,
)
from miles.utils.event_logger.models import LocalWeightChecksumEvent, OptimizerStateInfo


def _make_event(
    step: int,
    rank: int,
    param_hashes: dict[str, str] | None = None,
    buffer_hashes: dict[str, str] | None = None,
    optimizer_state_dict: dict | None = None,
) -> LocalWeightChecksumEvent:
    return LocalWeightChecksumEvent(
        step=step,
        rank=rank,
        param_hashes=param_hashes or {},
        buffer_hashes=buffer_hashes or {},
        optimizer_hashes=[
            OptimizerStateInfo(
                param_names={0: "pp0.weight"},
                state_dict=optimizer_state_dict or {},
            ),
        ] if optimizer_state_dict is not None else [],
    )


class TestCheck:
    def test_matching_replicas_no_mismatches(self) -> None:
        events = [
            _make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa"}),
            _make_event(step=0, rank=1, param_hashes={"pp0.weight": "aaa"}),
            _make_event(step=0, rank=2, param_hashes={"pp0.weight": "aaa"}),
        ]
        assert check(events) == []

    def test_param_hash_mismatch_detected(self) -> None:
        events = [
            _make_event(step=5, rank=0, param_hashes={"pp0.weight": "aaa"}),
            _make_event(step=5, rank=1, param_hashes={"pp0.weight": "zzz"}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        keys = [m.key for m in mismatches]
        assert any("param_hashes.pp0.weight" in k for k in keys)

    def test_missing_key_in_one_replica_detected(self) -> None:
        events = [
            _make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}),
            _make_event(step=0, rank=1, param_hashes={"pp0.weight": "aaa"}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        keys = [m.key for m in mismatches]
        assert any("pp0.bias" in k for k in keys)
        assert any("<missing>" in m.value_b for m in mismatches)

    def test_multiple_steps_only_mismatched_step_reported(self) -> None:
        events = [
            # Step 0: match
            _make_event(step=0, rank=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(step=0, rank=1, param_hashes={"pp0.w": "aaa"}),
            # Step 1: mismatch
            _make_event(step=1, rank=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(step=1, rank=1, param_hashes={"pp0.w": "zzz"}),
            # Step 2: match
            _make_event(step=2, rank=0, param_hashes={"pp0.w": "aaa"}),
            _make_event(step=2, rank=1, param_hashes={"pp0.w": "aaa"}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        assert all("step1" in m.value_a or "step1" in m.value_b for m in mismatches)

    def test_empty_events_no_mismatches(self) -> None:
        assert check([]) == []

    def test_buffer_mismatch_detected(self) -> None:
        events = [
            _make_event(step=0, rank=0, buffer_hashes={"pp0.running_mean": "aaa"}),
            _make_event(step=0, rank=1, buffer_hashes={"pp0.running_mean": "bbb"}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        assert any("buffer_hashes" in m.key for m in mismatches)

    def test_optimizer_state_mismatch_detected(self) -> None:
        events = [
            _make_event(step=3, rank=0, optimizer_state_dict={"state": {0: {"exp_avg": "aaa"}}}),
            _make_event(step=3, rank=1, optimizer_state_dict={"state": {0: {"exp_avg": "bbb"}}}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        assert any("exp_avg" in m.key for m in mismatches)

    def test_non_tensor_state_mismatch_detected(self) -> None:
        events = [
            _make_event(step=0, rank=0, optimizer_state_dict={"state": {0: {"step": 10}}}),
            _make_event(step=0, rank=1, optimizer_state_dict={"state": {0: {"step": 20}}}),
        ]
        mismatches = check(events)

        assert len(mismatches) >= 1
        assert any("step" in m.key for m in mismatches)


class TestFlattenEvent:
    def test_excludes_metadata_fields(self) -> None:
        event = _make_event(step=0, rank=0, param_hashes={"pp0.w": "aaa"})
        flat = _flatten_event(event)

        assert not any(k.startswith("step") or k.startswith("rank") or k.startswith("type") for k in flat.keys())
        assert "param_hashes.pp0.w" in flat

    def test_includes_optimizer_hashes(self) -> None:
        event = _make_event(step=0, rank=0, optimizer_state_dict={"state": {0: {"exp_avg": "hash1"}}})
        flat = _flatten_event(event)

        assert any("exp_avg" in k for k in flat.keys())


class TestFlattenNested:
    def test_flat_dict_with_string_values(self) -> None:
        result = _flatten_nested({"a": "hash1", "b": "hash2"}, prefix="root")
        assert result == {"root.a": "hash1", "root.b": "hash2"}

    def test_nested_dict(self) -> None:
        result = _flatten_nested({"state": {0: {"exp_avg": "h1"}}}, prefix="opt0")
        assert result == {"opt0.state.0.exp_avg": "h1"}

    def test_list_values(self) -> None:
        result = _flatten_nested({"params": ["a", "b"]}, prefix="opt0")
        assert result == {"opt0.params[0]": "a", "opt0.params[1]": "b"}

    def test_keeps_int_and_float_leaves(self) -> None:
        result = _flatten_nested({"lr": 0.001, "step": 42, "hash": "abc"}, prefix="root")
        assert result == {"root.hash": "abc", "root.lr": 0.001, "root.step": 42}

    def test_empty_prefix(self) -> None:
        result = _flatten_nested({"a": "x"}, prefix="")
        assert result == {"a": "x"}

    def test_empty_dict(self) -> None:
        result = _flatten_nested({}, prefix="root")
        assert result == {}
