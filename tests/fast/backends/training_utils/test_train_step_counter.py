"""Logical counter ownership, snapshot isolation and legacy checkpoint resume."""

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.train_step_counter import (
    TrainStepCounter,
    counter_for,
    restored_train_step,
    write_train_step_sidecar,
)


def test_variable_rollout_step_counts_and_independent_roles():
    actor, critic = SimpleNamespace(), SimpleNamespace()
    ids = [counter_for(actor).take() for steps in [3, 1, 2] for _ in range(steps)]
    assert ids == list(range(6))
    assert counter_for(critic).take() == 0
    assert counter_for(actor).next_step == 6


@pytest.mark.parametrize("invalid", [None, "bad", -1, True, 3.5])
def test_missing_or_invalid_state_resets_instead_of_retaining_stale_count(tmp_path, invalid, caplog):
    owner = SimpleNamespace(train_step_counter=TrainStepCounter(99))
    owner.train_step_counter = TrainStepCounter(restored_train_step(invalid, checkpoint_description="checkpoint"))
    assert counter_for(owner).take() == 0
    assert "cannot be inferred" in caplog.text


def test_legacy_sidecar_round_trip_and_embedded_metadata_precedence(tmp_path):
    path = tmp_path / "train_step_counter.txt"
    write_train_step_sidecar(path, 17)
    assert restored_train_step(None, legacy_paths=[path]) == 17
    assert restored_train_step(23, legacy_paths=[path]) == 23
    assert restored_train_step(0, legacy_paths=[path]) == 0
    path.write_text("corrupt")
    assert restored_train_step(None, legacy_paths=[path]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train_step_counter.txt"]


@pytest.fixture
def checkpoint_adapter(monkeypatch):
    """Load the real small adapter against Megatron's dependency boundary.

    This isolates metadata handling from unavailable Megatron/TE GPU packages;
    it does not simulate model or optimizer checkpoint correctness.
    """
    checkpointing = ModuleType("megatron.training.checkpointing")
    checkpointing.generate_state_dict = lambda args: {"args": args, "iteration": 4, "model": {"weight": torch.ones(1)}}
    checkpointing._load_base_checkpoint = lambda *a, **k: None
    megatron = ModuleType("megatron")
    training = ModuleType("megatron.training")
    training.checkpointing = checkpointing
    megatron.training = training
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/train_step_checkpoint.py"
    spec = importlib.util.spec_from_file_location("_test_step_checkpoint_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, checkpointing


def test_megatron_snapshot_copies_args_and_survives_serialization(checkpoint_adapter):
    module, backend = checkpoint_adapter
    args = SimpleNamespace(existing="unchanged")
    original = backend.generate_state_dict
    with module.checkpoint_train_step(7):
        snapshot = backend.generate_state_dict(args)
    with module.checkpoint_train_step(12):
        later = backend.generate_state_dict(args)
    assert backend.generate_state_dict is original
    assert vars(args) == {"existing": "unchanged"}
    assert snapshot["args"].miles_train_step == 7
    assert later["args"].miles_train_step == 12
    stream = io.BytesIO()
    torch.save(snapshot, stream)
    stream.seek(0)
    assert torch.load(stream, weights_only=False)["args"].miles_train_step == 7


@pytest.mark.parametrize("kind", ["GLOBAL", "LOCAL"])
def test_megatron_restores_metadata_from_selected_checkpoint(checkpoint_adapter, kind, tmp_path):
    module, backend = checkpoint_adapter
    checkpoint = {"args": SimpleNamespace(miles_train_step=11)}
    selected = tmp_path / "iter_0000004"
    backend._load_base_checkpoint = lambda *a, **k: (checkpoint, str(selected), False, SimpleNamespace(name=kind))
    original = backend._load_base_checkpoint
    with module.capture_loaded_train_step() as loaded:
        # Megatron may read once for format detection, then for actual loading.
        backend._load_base_checkpoint("unused-configured-path", rank0=True)
        backend._load_base_checkpoint("unused-configured-path", rank0=False)
    assert backend._load_base_checkpoint is original
    model = [SimpleNamespace(train_step_counter=TrainStepCounter(99))]
    module.restore_model_train_step(model, loaded)
    assert counter_for(model[0]).take() == 11
    assert (loaded["directory"] is None) == (kind == "LOCAL")


def test_megatron_legacy_restore_uses_selected_disk_path(checkpoint_adapter, tmp_path):
    module, backend = checkpoint_adapter
    directory = tmp_path / "selected" / "iter_0000004"
    directory.mkdir(parents=True)
    write_train_step_sidecar(directory / "train_step_counter.txt", 13)
    filename = directory / "mp_rank_00" / "model_optim_rng.pt"
    backend._load_base_checkpoint = lambda *a, **k: (
        {"args": SimpleNamespace()},
        str(filename),
        False,
        SimpleNamespace(name="LEGACY"),
    )
    with module.capture_loaded_train_step() as loaded:
        backend._load_base_checkpoint("different-directory")
    model = [SimpleNamespace()]
    module.restore_model_train_step(model, loaded)
    assert counter_for(model[0]).take() == 13
    module.restore_model_train_step(model, loaded, finetune=True)
    assert counter_for(model[0]).take() == 0


def test_missing_local_metadata_never_borrows_a_disk_counter(checkpoint_adapter, tmp_path):
    module, backend = checkpoint_adapter
    write_train_step_sidecar(tmp_path / "train_step_counter.txt", 500)
    backend._load_base_checkpoint = lambda *a, **k: (
        {"args": SimpleNamespace()},
        str(tmp_path),
        False,
        SimpleNamespace(name="LOCAL"),
    )
    with module.capture_loaded_train_step() as loaded:
        backend._load_base_checkpoint(str(tmp_path))
    model = [SimpleNamespace(train_step_counter=TrainStepCounter(99))]
    module.restore_model_train_step(model, loaded)
    assert counter_for(model[0]).take() == 0


def test_checkpoint_adapters_restore_hooks_when_save_or_load_fails(checkpoint_adapter):
    module, backend = checkpoint_adapter
    generator, loader = backend.generate_state_dict, backend._load_base_checkpoint
    with pytest.raises(RuntimeError):
        with module.checkpoint_train_step(4):
            raise RuntimeError("save failed")
    with pytest.raises(RuntimeError):
        with module.capture_loaded_train_step():
            raise RuntimeError("load failed")
    assert backend.generate_state_dict is generator
    assert backend._load_base_checkpoint is loader


def _fsdp_checkpoint():
    # Loading this module directly avoids the GPU-only actor package initializer.
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/checkpoint.py"
    spec = importlib.util.spec_from_file_location("_test_fsdp_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("legacy", [False, True])
def test_fsdp_restore_counter_and_rollout_position(monkeypatch, tmp_path, legacy):
    module = _fsdp_checkpoint()
    monkeypatch.setattr(module.dist, "barrier", lambda: None)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    metadata = {"next_rollout_id": 4, "micro_step": 20, "global_step": 17, "train_step_counter_version": 1}
    if legacy:
        metadata.pop("train_step_counter_version")
        metadata["global_step"] = 0  # old upstream field was never advanced
        directory = tmp_path / "iter_0000004"
        directory.mkdir()
        write_train_step_sidecar(directory / "train_step_counter.txt", 17)
    actor = SimpleNamespace(
        args=SimpleNamespace(load=str(tmp_path), start_rollout_id=None), global_step=99, micro_step=0
    )
    module.finalize_load(actor, {"iteration": 4, "metadata": metadata})
    assert actor.global_step == 17
    assert actor.micro_step == 20
    assert actor.args.start_rollout_id == 4
    module.finalize_load(actor, None)
    assert actor.global_step == 0


def test_fsdp_save_records_counter_with_correct_checkpoint_iteration(monkeypatch, tmp_path):
    module = _fsdp_checkpoint()
    monkeypatch.setattr(module.dist, "barrier", lambda: None)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module.torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(module.dcp, "save", lambda *a, **k: None)
    actor = SimpleNamespace(
        args=SimpleNamespace(save=str(tmp_path), no_save_optim=True), model=object(), global_step=17, micro_step=20
    )
    module.save(actor, 3)
    metadata = module._read_checkpoint_metadata(tmp_path / "iter_0000004/meta.json")
    assert metadata["global_step"] == 17
    assert metadata["train_step_counter_version"] == 1
    assert metadata["next_rollout_id"] == 4


@pytest.mark.parametrize("restore", [True, False])
def test_megatron_loader_uses_selected_adapter_counter_only_for_training_restore(
    checkpoint_adapter, tmp_path, restore
):
    from tests.fast.backends.training_utils.test_train_step_callers import _load_function

    module, backend = checkpoint_adapter
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    write_train_step_sidecar(adapter_dir / "train_step_counter.txt", 12)
    backend._load_base_checkpoint = lambda *a, **k: (
        {"args": SimpleNamespace(miles_train_step=7)},
        str(tmp_path),
        False,
        SimpleNamespace(name="GLOBAL"),
    )

    def load_base(**kwargs):
        backend._load_base_checkpoint(str(tmp_path))
        return 3, 0

    args = SimpleNamespace(load=str(tmp_path), lora_adapter_path=str(adapter_dir), finetune=False)
    namespace = {
        "Path": Path,
        "get_args": lambda: args,
        "logger": module.logging if hasattr(module, "logging") else SimpleNamespace(info=lambda *a: None),
        "_is_dir_nonempty": lambda path: True,
        "_is_megatron_checkpoint": lambda path: True,
        "is_dsv4_model": lambda args: False,
        "capture_loaded_train_step": module.capture_loaded_train_step,
        "_load_checkpoint_megatron": load_base,
        "restore_model_train_step": module.restore_model_train_step,
        "is_lora_enabled": lambda args: True,
        "load_lora_adapter": lambda *a, **k: (True, 5),
        "restored_train_step": restored_train_step,
        "TrainStepCounter": TrainStepCounter,
    }
    load = _load_function("miles/backends/megatron_utils/checkpoint.py", "load_checkpoint", namespace)
    model = [SimpleNamespace(train_step_counter=TrainStepCounter(99))]
    assert load(model, None, None, None, False, restore_train_step=restore) == (5, 0)
    assert counter_for(model[0]).next_step == (12 if restore else 99)


def test_lora_save_places_counter_with_adapter_checkpoint(tmp_path):
    from tests.fast.backends.training_utils.test_train_step_callers import _load_function

    namespace = {
        "Path": Path,
        "get_args": lambda: SimpleNamespace(save=str(tmp_path)),
        "logger": SimpleNamespace(info=lambda *a: None),
        "is_lora_model": lambda model: True,
        "save_lora_checkpoint": lambda model, args, path, **kwargs: Path(path).mkdir(parents=True),
        "is_first_replica_megatron_main_rank": lambda: True,
        "write_train_step_sidecar": write_train_step_sidecar,
        "counter_for": counter_for,
    }
    save = _load_function("miles/backends/megatron_utils/checkpoint.py", "save_checkpoint_with_lora", namespace)
    save(4, [SimpleNamespace(train_step_counter=TrainStepCounter(19))], None, None)
    assert (tmp_path / "iter_0000004/adapter/train_step_counter.txt").read_text() == "19"


def test_megatron_save_attaches_count_and_preserves_recovery_context(checkpoint_adapter):
    from tests.fast.backends.training_utils.test_train_step_callers import _load_function

    module, backend = checkpoint_adapter
    args = SimpleNamespace(ci_test=False)
    captured = {}

    def save_backend(*positional, **kwargs):
        captured.update(kwargs)
        captured["snapshot"] = backend.generate_state_dict(args)

    namespace = {
        "get_args": lambda: args,
        "should_disable_forward_pre_hook": lambda args: False,
        "clear_memory": lambda: None,
        "is_lora_model": lambda model: False,
        "checkpoint_train_step": module.checkpoint_train_step,
        "counter_for": counter_for,
        "save_checkpoint": save_backend,
        "preprocess_common_state_dict": lambda state: state,
    }
    save = _load_function("miles/backends/megatron_utils/model.py", "save", namespace)
    context = {"local_checkpoint_manager": object()}
    save(4, [SimpleNamespace(train_step_counter=TrainStepCounter(19))], None, None, context, True)
    assert captured["snapshot"]["args"].miles_train_step == 19
    assert captured["checkpointing_context"] is context
    assert captured["non_persistent_ckpt"] is True
    assert captured["preprocess_common_state_dict_fn"] is namespace["preprocess_common_state_dict"]
