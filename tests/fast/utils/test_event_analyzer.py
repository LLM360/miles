"""Tests for event_analyzer rules and analyzer."""

from pathlib import Path

from miles.utils.event_analyzer.analyzer import run_analysis
from miles.utils.event_logger.logger import EventLogger
from miles.utils.event_logger.models import LocalWeightChecksumEvent
from miles.utils.process_identity import MainProcessIdentity


def _log_checksum(
    event_logger: EventLogger,
    step: int,
    rank: int,
    param_hashes: dict[str, str] | None = None,
    buffer_hashes: dict[str, str] | None = None,
    master_param_hashes: dict[str, str] | None = None,
    optimizer_state_hashes: dict[str, str] | None = None,
) -> None:
    event_logger.log(LocalWeightChecksumEvent(
        step=step,
        rank=rank,
        param_hashes=param_hashes or {},
        buffer_hashes=buffer_hashes or {},
        master_param_hashes=master_param_hashes or {},
        optimizer_state_hashes=optimizer_state_hashes or {},
    ))


class TestRunAnalysisWeightChecksum:
    def test_matching_checksums_across_replicas_passes(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log_checksum(event_logger, step=0, rank=0, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"})
        _log_checksum(event_logger, step=0, rank=1, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"})
        _log_checksum(event_logger, step=0, rank=2, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"})
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)
        assert mismatches == []

    def test_param_hash_mismatch_reports_correct_details(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log_checksum(event_logger, step=5, rank=0, param_hashes={"pp0.weight": "aaa"})
        _log_checksum(event_logger, step=5, rank=1, param_hashes={"pp0.weight": "zzz"})
        _log_checksum(event_logger, step=5, rank=2, param_hashes={"pp0.weight": "aaa"})
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        m = mismatches[0]
        assert m.step == 5
        assert m.tensor_category == "param"
        assert m.tensor_name == "pp0.weight"
        assert 0 in m.cell_indices
        assert 1 in m.cell_indices

    def test_missing_tensor_in_one_replica_reports_mismatch(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log_checksum(event_logger, step=0, rank=0, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"})
        _log_checksum(event_logger, step=0, rank=1, param_hashes={"pp0.weight": "aaa"})
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        m = mismatches[0]
        assert m.tensor_name == "pp0.bias"
        assert "<missing>" in m.hashes

    def test_multiple_steps_only_reports_mismatched_step(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())

        # Step 0: all match
        _log_checksum(event_logger, step=0, rank=0, param_hashes={"pp0.weight": "aaa"})
        _log_checksum(event_logger, step=0, rank=1, param_hashes={"pp0.weight": "aaa"})

        # Step 1: mismatch
        _log_checksum(event_logger, step=1, rank=0, param_hashes={"pp0.weight": "aaa"})
        _log_checksum(event_logger, step=1, rank=1, param_hashes={"pp0.weight": "zzz"})

        # Step 2: all match
        _log_checksum(event_logger, step=2, rank=0, param_hashes={"pp0.weight": "aaa"})
        _log_checksum(event_logger, step=2, rank=1, param_hashes={"pp0.weight": "aaa"})
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].step == 1

    def test_empty_directory_returns_no_mismatches(self, tmp_path: Path) -> None:
        mismatches = run_analysis(event_dir=tmp_path)
        assert mismatches == []

    def test_buffer_mismatch_detected(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log_checksum(event_logger, step=0, rank=0, buffer_hashes={"pp0.running_mean": "aaa"})
        _log_checksum(event_logger, step=0, rank=1, buffer_hashes={"pp0.running_mean": "bbb"})
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].tensor_category == "buffer"

    def test_optimizer_state_mismatch_detected(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        _log_checksum(event_logger, step=3, rank=0, optimizer_state_hashes={"pp0.weight/exp_avg": "aaa"})
        _log_checksum(event_logger, step=3, rank=1, optimizer_state_hashes={"pp0.weight/exp_avg": "bbb"})
        event_logger.close()

        mismatches = run_analysis(event_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].tensor_category == "optimizer_state"
        assert mismatches[0].tensor_name == "pp0.weight/exp_avg"
