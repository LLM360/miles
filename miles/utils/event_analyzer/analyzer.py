"""Centralized event analyzer that reads events and runs all rules."""

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

from miles.utils.event_analyzer.rules import cross_replica_weight_checksum, witness as witness_rule
from miles.utils.event_logger.logger import read_events

logger = logging.getLogger(__name__)


def run_analysis_from_args(args: Namespace) -> None:
    if not getattr(args, "enable_event_analyzer", False):
        return

    event_dir = getattr(args, "save_debug_event_data", None)
    if event_dir is None:
        return

    issues = run_analysis(event_dir=Path(event_dir))

    # Fail fast, we want to stop the system if sanity check fails
    if issues:
        raise ValueError(f"Event analysis found issues: {issues}")


def run_analysis(event_dir: Path) -> list[Any]:
    events = read_events(event_dir)
    if not events:
        return []

    issues = cross_replica_weight_checksum.check(events)

    witness_mismatches = witness_rule.check(events)
    for m in witness_mismatches:
        logger.error("Witness mismatch at step %d quorum %d: %s", m.step, m.quorum_id, m.description)
    issues.extend(witness_mismatches)

    return issues
