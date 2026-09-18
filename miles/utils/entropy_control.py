"""Entropy-control helpers shared by the RL training backends.

The adaptive clipping rule follows MAI-Thinking-1, Section 3.1.1.  MILES
stores clip ranges as offsets around one, so the paper's upper ratio bound

    (1 - epsilon) ** -1 + k

is represented by ``eps_clip_high = (1 - epsilon) ** -1 - 1 + k``.
"""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path


logger = logging.getLogger(__name__)

_STATE_FILE = "adaptive_clip_state.json"


def adaptive_clip_high(eps_clip: float, relaxation: float) -> float:
    """Return MILES' upper clip offset for MAI's log-symmetric interval."""
    if not 0 <= eps_clip < 1:
        raise ValueError(f"eps_clip must be in [0, 1), got {eps_clip}")
    if relaxation < 0:
        raise ValueError(f"adaptive clip relaxation must be non-negative, got {relaxation}")
    return 1.0 / (1.0 - eps_clip) - 1.0 + relaxation


def update_adaptive_clip_relaxation(
    relaxation: float,
    estimated_entropy: float,
    target_entropy: float,
    step_size: float,
    max_relaxation: float,
) -> float:
    """Apply MAI's sign-integral controller update to ``k``."""
    if estimated_entropy < target_entropy:
        direction = 1.0
    elif estimated_entropy > target_entropy:
        direction = -1.0
    else:
        direction = 0.0
    return min(max(relaxation + step_size * direction, 0.0), max_relaxation)


def initialize_adaptive_clip(args: Namespace) -> None:
    """Initialize the runtime controller state from command-line arguments."""
    if not getattr(args, "use_adaptive_clip", False):
        return
    args.adaptive_clip_relaxation = float(args.adaptive_clip_initial_relaxation)
    args.eps_clip_high = adaptive_clip_high(args.eps_clip, args.adaptive_clip_relaxation)


def _checkpoint_state_path(checkpoint_dir: str | None, iteration: int | None) -> Path | None:
    if checkpoint_dir is None or iteration is None:
        return None
    base = Path(checkpoint_dir).expanduser()
    iteration_dir = f"iter_{int(iteration):07d}"
    return (base if base.name == iteration_dir else base / iteration_dir) / _STATE_FILE


def save_adaptive_clip_state(args: Namespace, checkpoint_dir: str | None, iteration: int | None) -> None:
    """Persist controller state next to a model checkpoint."""
    if not getattr(args, "use_adaptive_clip", False):
        return
    path = _checkpoint_state_path(checkpoint_dir, iteration)
    if path is None:
        return
    payload = {
        "relaxation": float(args.adaptive_clip_relaxation),
        "eps_clip_high": float(args.eps_clip_high),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary_path.replace(path)
    except OSError as error:
        logger.warning("Failed to persist adaptive-clip state: %s", error)


def load_adaptive_clip_state(args: Namespace, checkpoint_dir: str | None, iteration: int | None) -> None:
    """Restore controller state, retaining the configured initial state if absent."""
    if not getattr(args, "use_adaptive_clip", False):
        return
    path = _checkpoint_state_path(checkpoint_dir, iteration)
    if path is None or not path.is_file():
        return
    try:
        payload = json.loads(path.read_text())
        relaxation = float(payload["relaxation"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.warning("Failed to restore adaptive-clip state from %s: %s", path, error)
        return

    args.adaptive_clip_relaxation = min(max(relaxation, 0.0), args.adaptive_clip_max_relaxation)
    args.eps_clip_high = adaptive_clip_high(args.eps_clip, args.adaptive_clip_relaxation)
