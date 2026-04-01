"""Tests for event_analyzer/analyzer.py."""

from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.event_analyzer.analyzer import run_analysis, run_analysis_from_args
from miles.utils.event_logger.logger import EventLogger
from miles.utils.event_logger.models import LocalWeightChecksumEvent, LocalWeightChecksumState
from miles.utils.process_identity import MainProcessIdentity


def _make_event(
    step: int,
    rank: int,
    param_hashes: dict[str, str] | None = None,
) -> LocalWeightChecksumEvent:
    return LocalWeightChecksumEvent(
        step=step,
        cell_index=0,
        rank_within_cell=rank,
        state=LocalWeightChecksumState(
            param_hashes=param_hashes or {},
            buffer_hashes={},
            optimizer_hashes=[],
        ),
    )


class TestRunAnalysis:
    def test_empty_directory_returns_no_issues(self, tmp_path: Path) -> None:
        assert run_analysis(event_dir=tmp_path) == []

    def test_delegates_to_rules_and_returns_issues(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        event_logger.log(_make_event(step=0, rank=0, param_hashes={"pp0.w": "aaa"}))
        event_logger.log(_make_event(step=0, rank=1, param_hashes={"pp0.w": "zzz"}))
        event_logger.close()

        issues = run_analysis(event_dir=tmp_path)
        assert len(issues) == 1


class TestRunAnalysisFromArgs:
    def test_skips_when_disabled(self) -> None:
        args = Namespace(enable_event_analyzer=False, save_debug_event_data="/tmp/whatever")
        run_analysis_from_args(args)

    def test_skips_when_no_event_dir(self) -> None:
        args = Namespace(enable_event_analyzer=True)
        run_analysis_from_args(args)

    def test_raises_on_mismatch(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        event_logger.log(_make_event(step=0, rank=0, param_hashes={"pp0.w": "aaa"}))
        event_logger.log(_make_event(step=0, rank=1, param_hashes={"pp0.w": "zzz"}))
        event_logger.close()

        args = Namespace(enable_event_analyzer=True, save_debug_event_data=str(tmp_path))
        with pytest.raises(ValueError, match="issues"):
            run_analysis_from_args(args)

    def test_passes_when_all_match(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        event_logger.log(_make_event(step=0, rank=0, param_hashes={"pp0.w": "aaa"}))
        event_logger.log(_make_event(step=0, rank=1, param_hashes={"pp0.w": "aaa"}))
        event_logger.close()

        args = Namespace(enable_event_analyzer=True, save_debug_event_data=str(tmp_path))
        run_analysis_from_args(args)
