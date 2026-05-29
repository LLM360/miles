"""Trainer-owned logical step IDs and compatibility with stable sidecars."""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainStepCounter:
    next_step: int = 0

    def take(self) -> int:
        step = self.next_step
        self.next_step += 1
        return step


def counter_for(owner) -> TrainStepCounter:
    """Each trainer/model role owns its counter; logging has no mutable state."""
    if not isinstance(getattr(owner, "train_step_counter", None), TrainStepCounter):
        owner.train_step_counter = TrainStepCounter()
    return owner.train_step_counter


def valid_train_step(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        step = int(value)
    except ValueError:
        return None
    return step if step >= 0 else None


def read_train_step_sidecar(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return valid_train_step(path.read_text().strip())
    except OSError:
        return None


def restored_train_step(value, *, legacy_paths=(), checkpoint_description=None) -> int:
    step = valid_train_step(value)
    for path in legacy_paths:
        if step is not None:
            break
        step = read_train_step_sidecar(path)
    if step is None:
        if checkpoint_description is not None:
            logger.warning(
                "No valid training-step count in %s; resetting to 0. Variable historical step counts cannot be inferred.",
                checkpoint_description,
            )
        return 0
    return step


def write_train_step_sidecar(path: Path, next_step: int) -> None:
    """Atomic legacy-compatible metadata write alongside an existing checkpoint."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(str(next_step))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        logger.warning("Failed to persist training-step count at %s", path, exc_info=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
