"""Capture step metadata through Megatron's actual checkpoint selection.

Megatron's public loader returns only iteration/FLOPs. These scoped adapters
capture the metadata it already reads, without loading weights again or guessing
whether disk, a local manager, or an in-memory checkpoint was selected. They do
not alter model/optimizer state or checkpoint selection. Actor calls are serial.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from functools import wraps
from pathlib import Path

from megatron.training import checkpointing

from miles.backends.training_utils.train_step_counter import TrainStepCounter, restored_train_step

_STEP_ATTRIBUTE = "miles_train_step"


@contextmanager
def checkpoint_train_step(next_step: int):
    """Snapshot a scalar in existing checkpoint args, including local saves.

    Copy args so an asynchronous save or an in-memory snapshot never observes a
    later counter value, and the live configuration is never mutated.
    """
    original = checkpointing.generate_state_dict

    @wraps(original)
    def generate_with_step(*args, **kwargs):
        state = original(*args, **kwargs)
        snapshot_args = copy(state["args"])
        setattr(snapshot_args, _STEP_ATTRIBUTE, next_step)
        return {**state, "args": snapshot_args}

    checkpointing.generate_state_dict = generate_with_step
    try:
        yield
    finally:
        checkpointing.generate_state_dict = original


@contextmanager
def capture_loaded_train_step():
    original = checkpointing._load_base_checkpoint
    loaded = {"value": None, "directory": None, "description": None}

    @wraps(original)
    def load_with_step(*args, **kwargs):
        result = original(*args, **kwargs)
        state, name, _release, kind = result
        if isinstance(state, Mapping) and state:
            loaded["value"] = getattr(state.get("args"), _STEP_ATTRIBUTE, None)
            loaded["description"] = name or "selected local checkpoint"
            # A LOCAL checkpoint is held by its manager; never use args.load's
            # unrelated disk sidecar, even when both checkpoints share an ID.
            if getattr(kind, "name", None) == "LOCAL":
                loaded["directory"] = None
            elif name is not None:
                path = Path(name)
                loaded["directory"] = path.parent.parent if path.name.endswith(".pt") else path
        return result

    checkpointing._load_base_checkpoint = load_with_step
    try:
        yield loaded
    finally:
        checkpointing._load_base_checkpoint = original


def restore_model_train_step(model, loaded, *, finetune=False):
    directory = loaded["directory"]
    legacy_paths = [directory / "train_step_counter.txt"] if directory is not None else []
    step = (
        0
        if finetune
        else restored_train_step(
            loaded["value"],
            legacy_paths=legacy_paths,
            checkpoint_description=loaded["description"],
        )
    )
    model[0].train_step_counter = TrainStepCounter(step)
