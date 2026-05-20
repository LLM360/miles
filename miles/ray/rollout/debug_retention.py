"""Prune only generated training rollout dumps in an expired rollout-ID window."""

import glob
import logging
import re
from pathlib import Path
from string import Formatter

logger = logging.getLogger(__name__)


def prune_rollout_dumps(path_template: str, *, trajectory_template: str | None, rollout_id: int, retain: int):
    if retain <= 0:
        return
    cutoff = rollout_id - retain
    templates = {
        path_template,
        str(Path(path_template).with_suffix(".pt")),
        str(Path(path_template).with_suffix(".parquet")),
    }
    try:
        patterns = [(template, *_training_dump_pattern(template)) for template in templates]
    except ValueError as error:
        logger.warning("Skipping rollout retention for %s: %s", path_template, error)
        return
    for template, pattern, regex in patterns:
        for name in glob.iglob(pattern):
            match = regex.fullmatch(name)
            if match is None or int(match["rollout_id"]) > cutoff:
                continue
            old_id = match["rollout_id"]
            # Verify the full configured name before deleting; never match eval IDs.
            if template.format(rollout_id=old_id) != name:
                continue
            path = Path(name)
            sidecars = [path.parent.parent / "dashboard_columns" / f"rollout_{old_id}.parquet"]
            # A shared trajectory filename cannot be attributed to an expired ID.
            if trajectory_template and "{rollout_id}" in trajectory_template:
                sidecars.append(Path(trajectory_template.format(rollout_id=old_id)))
            for expired in [path, *sidecars]:
                if expired.is_file():
                    expired.unlink()
                    logger.info("Removed expired rollout dump %s", expired)


def _training_dump_pattern(template: str) -> tuple[str, re.Pattern]:
    pattern, regex = "", ""
    seen = False
    for literal, field, spec, conversion in Formatter().parse(template):
        pattern += glob.escape(literal)
        regex += re.escape(literal)
        if field is None:
            continue
        if field != "rollout_id" or spec or conversion:
            raise ValueError("Rollout retention requires a plain {rollout_id} placeholder")
        pattern += "*"
        regex += r"(?P=rollout_id)" if seen else r"(?P<rollout_id>[0-9]+)"
        seen = True
    if not seen:
        raise ValueError("Rollout retention requires a {rollout_id} placeholder")
    return pattern, re.compile(regex)
