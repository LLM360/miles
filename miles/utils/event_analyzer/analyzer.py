"""Centralized event analyzer that reads events and runs all rules."""

import logging
from argparse import Namespace
from pathlib import Path

from miles.utils.event_analyzer.rules.weight_checksum import ChecksumMismatch, check_weight_checksums
from miles.utils.event_logger.logger import read_events

logger = logging.getLogger(__name__)


def run_analysis_from_args(args: Namespace) -> None:
    """Run event analysis if event logging is enabled. Safe to call unconditionally."""
    event_dir = getattr(args, "save_debug_event_data", None)
    if event_dir is None:
        return

    mismatches = run_analysis(event_dir=Path(event_dir))

    # Fail fast, we want to stop the system if sanity check fails
    if mismatches:
        raise ValueError(f"Event analysis found {len(mismatches)} mismatches, see above for details")


def run_analysis(event_dir: Path) -> list[ChecksumMismatch]:
    """Read all events from event_dir and run all analysis rules.

    Returns:
        List of mismatches found. Empty list means all replicas match.
    """
    events = read_events(event_dir)
    if not events:
        return []

    mismatches = check_weight_checksums(events)

    for m in mismatches:
        logger.error(
            "Weight checksum mismatch at step %d: %s/%s diverged across ranks %s",
            m.step,
            m.category,
            m.key,
            m.cell_indices,
        )

    return mismatches
