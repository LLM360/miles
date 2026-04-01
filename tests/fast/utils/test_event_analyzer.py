"""Tests for event_analyzer rules and analyzer."""

from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.event_analyzer.analyzer import run_analysis, run_analysis_from_args
from miles.utils.event_analyzer.rules.weight_checksum import _flatten_nested
from miles.utils.event_logger.logger import EventLogger
from miles.utils.event_logger.models import LocalWeightChecksumEvent, OptimizerStateInfo
from miles.utils.process_identity import MainProcessIdentity


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


def _log(event_logger: EventLogger, event: LocalWeightChecksumEvent) -> None:
    event_logger.log(event)


class TestRunAnalysisWeightChecksum:
    def test_matching_checksums_across_replicas_passes(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log(event_logger, _make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}))
        _log(event_logger, _make_event(step=0, rank=1, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}))
        _log(event_logger, _make_event(step=0, rank=2, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}))
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)
        assert mismatches == []

    def test_param_hash_mismatch_reports_correct_details(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log(event_logger, _make_event(step=5, rank=0, param_hashes={"pp0.weight": "aaa"}))
        _log(event_logger, _make_event(step=5, rank=1, param_hashes={"pp0.weight": "zzz"}))
        _log(event_logger, _make_event(step=5, rank=2, param_hashes={"pp0.weight": "aaa"}))
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        m = mismatches[0]
        assert m.step == 5
        assert m.category == "param"
        assert m.key == "pp0.weight"
        assert 0 in m.cell_indices
        assert 1 in m.cell_indices

    def test_missing_tensor_in_one_replica_reports_mismatch(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log(event_logger, _make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}))
        _log(event_logger, _make_event(step=0, rank=1, param_hashes={"pp0.weight": "aaa"}))
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].key == "pp0.bias"
        assert "<missing>" in mismatches[0].values

    def test_multiple_steps_only_reports_mismatched_step(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())

        # Step 0: all match
        _log(event_logger, _make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa"}))
        _log(event_logger, _make_event(step=0, rank=1, param_hashes={"pp0.weight": "aaa"}))

        # Step 1: mismatch
        _log(event_logger, _make_event(step=1, rank=0, param_hashes={"pp0.weight": "aaa"}))
        _log(event_logger, _make_event(step=1, rank=1, param_hashes={"pp0.weight": "zzz"}))

        # Step 2: all match
        _log(event_logger, _make_event(step=2, rank=0, param_hashes={"pp0.weight": "aaa"}))
        _log(event_logger, _make_event(step=2, rank=1, param_hashes={"pp0.weight": "aaa"}))
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].step == 1

    def test_empty_directory_returns_no_mismatches(self, tmp_path: Path) -> None:
        mismatches = run_analysis(event_dir=tmp_path)
        assert mismatches == []

    def test_buffer_mismatch_detected(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log(event_logger, _make_event(step=0, rank=0, buffer_hashes={"pp0.running_mean": "aaa"}))
        _log(event_logger, _make_event(step=0, rank=1, buffer_hashes={"pp0.running_mean": "bbb"}))
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].category == "buffer"

    def test_optimizer_state_mismatch_detected(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log(event_logger, _make_event(
            step=3, rank=0,
            optimizer_state_dict={"state": {0: {"exp_avg": "aaa"}}},
        ))
        _log(event_logger, _make_event(
            step=3, rank=1,
            optimizer_state_dict={"state": {0: {"exp_avg": "bbb"}}},
        ))
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].category == "optimizer"
        assert "exp_avg" in mismatches[0].key


class TestRunAnalysisFromArgs:
    def test_skips_when_no_event_dir(self) -> None:
        args = Namespace()
        run_analysis_from_args(args)

    def test_raises_on_mismatch(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log(event_logger, _make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa"}))
        _log(event_logger, _make_event(step=0, rank=1, param_hashes={"pp0.weight": "zzz"}))
        event_logger.close()

        args = Namespace(save_debug_event_data=str(tmp_path))
        with pytest.raises(ValueError, match="mismatches"):
            run_analysis_from_args(args)

    def test_passes_when_all_match(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log(event_logger, _make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa"}))
        _log(event_logger, _make_event(step=0, rank=1, param_hashes={"pp0.weight": "aaa"}))
        event_logger.close()

        args = Namespace(save_debug_event_data=str(tmp_path))
        run_analysis_from_args(args)


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

    def test_ignores_non_string_leaves(self) -> None:
        result = _flatten_nested({"lr": 0.001, "hash": "abc"}, prefix="root")
        assert result == {"root.hash": "abc"}

    def test_empty_dict(self) -> None:
        result = _flatten_nested({}, prefix="root")
        assert result == {}


class TestReadEventsMalformedJsonl:
    def test_malformed_line_skipped_with_warning(self, tmp_path: Path) -> None:
        from miles.utils.event_logger.logger import read_events

        jsonl_file = tmp_path / "events.jsonl"
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log(event_logger, _make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa"}))
        event_logger.close()

        # Append a malformed line
        with open(jsonl_file, "a") as f:
            f.write("this is not valid json\n")

        events = read_events(tmp_path)
        assert len(events) == 1
