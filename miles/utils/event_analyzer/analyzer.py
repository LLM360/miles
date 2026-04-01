"""Centralized event analyzer that reads events and runs all rules."""

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

from miles.utils.event_analyzer.rules.weight_checksum import ChecksumMismatchIssue, check_weight_checksums
from miles.utils.event_logger.logger import read_events

logger = logging.getLogger(__name__)


def run_analysis_from_args(args: Namespace) -> None:
    """Run event analysis if event logging is enabled. Safe to call unconditionally."""
    event_dir = getattr(args, "save_debug_event_data", None)
    if event_dir is None:
        return

    issues = run_analysis(event_dir=Path(event_dir))

    # Fail fast, we want to stop the system if sanity check fails
    if issues:
        raise ValueError(f"Event analysis found issues: {issues}")


def run_analysis(event_dir: Path) -> list[Any]:
    """Read all events from event_dir and run all analysis rules.

    Returns:
        List of issues found. Empty list means all replicas match.
    """
    events = read_events(event_dir)
    if not events:
        return []

    # TODO: more check rules
    issues = check_weight_checksums(events)

    return issues
