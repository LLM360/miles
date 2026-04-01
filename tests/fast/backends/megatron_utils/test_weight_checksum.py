"""Tests for weight_checksum module."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from miles.backends.megatron_utils.weight_checksum import (
    compute_and_dump_weight_checksums,
    _compute_weight_checksums,
)
from miles.utils.event_logger.logger import EventLogger, set_event_logger
from miles.utils.event_logger.models import LocalWeightChecksumEvent
from miles.utils.process_identity import MainProcessIdentity


def _make_mock_model_chunk(
    params: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor] | None = None
) -> MagicMock:
    """Create a mock DDP model chunk with given named parameters and buffers."""
    chunk = MagicMock()
    chunk.named_parameters.return_value = sorted(params.items(), key=lambda x: x[0])
    chunk.named_buffers.return_value = sorted((buffers or {}).items(), key=lambda x: x[0])
    return chunk


def _make_mock_optimizer(
    fp16_params: list[torch.nn.Parameter] | None = None,
    fp32_params: list[torch.nn.Parameter] | None = None,
    states: dict[torch.nn.Parameter, dict[str, torch.Tensor]] | None = None,
) -> MagicMock:
    """Create a mock optimizer with chained_optimizers supporting fp32 master weights."""
    optimizer = MagicMock()

    if fp16_params is None:
        optimizer.chained_optimizers = []
        return optimizer

    chained = MagicMock()
    chained.__class__ = type("Float16OptimizerWithFloat16Params", (), {})
    chained.float16_groups = [fp16_params]
    chained.fp32_from_float16_groups = [fp32_params or []]

    inner_optimizer = MagicMock()
    inner_optimizer.state = states or {}
    chained.optimizer = inner_optimizer

    optimizer.chained_optimizers = [chained]
    return optimizer


class TestComputeWeightChecksums:
    def test_param_hashes_keys_match_expected_names(self) -> None:
        params = {
            "module.layers.0.weight": torch.randn(4, 4),
            "module.layers.1.weight": torch.randn(4, 4),
        }
        model = [_make_mock_model_chunk(params=params)]
        optimizer = _make_mock_optimizer()

        entry = _compute_weight_checksums(model=model, optimizer=optimizer)

        assert set(entry.param_hashes.keys()) == {
            "pp0.module.layers.0.weight",
            "pp0.module.layers.1.weight",
        }

    def test_hash_determinism_same_params_same_hash(self) -> None:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        params = {"weight": tensor}
        model = [_make_mock_model_chunk(params=params)]
        optimizer = _make_mock_optimizer()

        entry1 = _compute_weight_checksums(model=model, optimizer=optimizer)
        entry2 = _compute_weight_checksums(model=model, optimizer=optimizer)

        assert entry1.param_hashes == entry2.param_hashes

    def test_param_value_change_changes_hash(self) -> None:
        tensor_a = torch.tensor([1.0, 2.0, 3.0])
        tensor_b = torch.tensor([1.0, 2.0, 4.0])
        model_a = [_make_mock_model_chunk(params={"weight": tensor_a})]
        model_b = [_make_mock_model_chunk(params={"weight": tensor_b})]
        optimizer = _make_mock_optimizer()

        entry_a = _compute_weight_checksums(model=model_a, optimizer=optimizer)
        entry_b = _compute_weight_checksums(model=model_b, optimizer=optimizer)

        assert entry_a.param_hashes["pp0.weight"] != entry_b.param_hashes["pp0.weight"]

    def test_buffer_hashing_produces_values(self) -> None:
        params = {"weight": torch.randn(4, 4)}
        buffers = {"running_mean": torch.randn(4), "running_var": torch.randn(4)}
        model = [_make_mock_model_chunk(params=params, buffers=buffers)]
        optimizer = _make_mock_optimizer()

        entry = _compute_weight_checksums(model=model, optimizer=optimizer)

        assert "pp0.running_mean" in entry.buffer_hashes
        assert "pp0.running_var" in entry.buffer_hashes
        assert len(entry.buffer_hashes["pp0.running_mean"]) == 64  # SHA-256 hex length

    @patch("miles.backends.megatron_utils.weight_checksum.Float16OptimizerWithFloat16Params", create=True)
    def test_master_param_hashing_with_mock_optimizer(self) -> None:
        fp16_param = torch.nn.Parameter(torch.randn(4, 4))
        fp16_param.main_param = MagicMock()
        fp16_param.main_param._param_name = "pp0.layers.0.weight"

        fp32_param = torch.nn.Parameter(torch.randn(4, 4))

        chained = MagicMock()
        chained.float16_groups = [[fp16_param]]
        chained.fp32_from_float16_groups = [[fp32_param]]
        chained.optimizer.state = {}

        optimizer = MagicMock()
        optimizer.chained_optimizers = [chained]

        model = [_make_mock_model_chunk(params={"layers.0.weight": torch.randn(4, 4)})]

        with patch("miles.backends.megatron_utils.weight_checksum.Float16OptimizerWithFloat16Params") as mock_cls:
            mock_cls.__instancecheck__ = lambda self, instance: instance is chained
            entry = _compute_weight_checksums(model=model, optimizer=optimizer)

        assert "pp0.layers.0.weight" in entry.master_param_hashes

    @patch("miles.backends.megatron_utils.weight_checksum.Float16OptimizerWithFloat16Params", create=True)
    def test_optimizer_state_hashing_exp_avg_and_exp_avg_sq(self) -> None:
        fp16_param = torch.nn.Parameter(torch.randn(4, 4))
        fp16_param.main_param = MagicMock()
        fp16_param.main_param._param_name = "pp0.weight"

        fp32_param = torch.nn.Parameter(torch.randn(4, 4))
        exp_avg = torch.randn(4, 4)
        exp_avg_sq = torch.randn(4, 4)

        chained = MagicMock()
        chained.float16_groups = [[fp16_param]]
        chained.fp32_from_float16_groups = [[fp32_param]]
        chained.optimizer.state = {fp32_param: {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}}

        optimizer = MagicMock()
        optimizer.chained_optimizers = [chained]

        model = [_make_mock_model_chunk(params={"weight": torch.randn(4, 4)})]

        with patch("miles.backends.megatron_utils.weight_checksum.Float16OptimizerWithFloat16Params") as mock_cls:
            mock_cls.__instancecheck__ = lambda self, instance: instance is chained
            entry = _compute_weight_checksums(model=model, optimizer=optimizer)

        assert "pp0.weight/exp_avg" in entry.optimizer_state_hashes
        assert "pp0.weight/exp_avg_sq" in entry.optimizer_state_hashes


class TestComputeAndDumpWeightChecksums:
    def test_does_nothing_when_disabled(self, tmp_path: Path) -> None:
        args = Namespace(save_local_weight_checksum=False)
        model = [_make_mock_model_chunk(params={})]
        optimizer = _make_mock_optimizer()

        compute_and_dump_weight_checksums(args=args, model=model, optimizer=optimizer, step=0)

    def test_dumps_when_enabled(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        set_event_logger(event_logger)
        try:
            args = Namespace(save_local_weight_checksum=True)
            model = [_make_mock_model_chunk(params={"w": torch.randn(2, 2)})]
            optimizer = _make_mock_optimizer()

            with patch("miles.backends.megatron_utils.weight_checksum.torch.distributed.get_rank", return_value=7):
                compute_and_dump_weight_checksums(args=args, model=model, optimizer=optimizer, step=4)

            event_logger.close()

            from miles.utils.event_logger.logger import read_events

            events = read_events(tmp_path)
            checksum_events = [e for e in events if isinstance(e, LocalWeightChecksumEvent)]
            assert len(checksum_events) == 1
            assert checksum_events[0].step == 4
            assert checksum_events[0].rank == 7
        finally:
            set_event_logger(None)
