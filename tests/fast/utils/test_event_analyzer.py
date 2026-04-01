"""Tests for event_analyzer post-hoc weight checksum checker."""

from pathlib import Path


from miles.backends.megatron_utils.weight_checksum import WeightChecksumEntry
from miles.utils.event_analyzer import check_weight_checksums


def _write_entry(base_dir: Path, step: int, rank: int, entry: WeightChecksumEntry) -> None:
    """Write a checksum entry to the expected file path."""
    file_path = base_dir / f"step_{step:07d}" / f"rank_{rank:04d}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(entry.model_dump_json(indent=2))


def _make_entry(
    param_hashes: dict[str, str] | None = None,
    buffer_hashes: dict[str, str] | None = None,
    master_param_hashes: dict[str, str] | None = None,
    optimizer_state_hashes: dict[str, str] | None = None,
) -> WeightChecksumEntry:
    return WeightChecksumEntry(
        param_hashes=param_hashes or {},
        buffer_hashes=buffer_hashes or {},
        master_param_hashes=master_param_hashes or {},
        optimizer_state_hashes=optimizer_state_hashes or {},
    )


class TestCheckWeightChecksums:
    def test_matching_checksums_across_replicas_passes(self, tmp_path: Path) -> None:
        entry = _make_entry(param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"})

        _write_entry(tmp_path, step=0, rank=0, entry=entry)
        _write_entry(tmp_path, step=0, rank=1, entry=entry)
        _write_entry(tmp_path, step=0, rank=2, entry=entry)

        mismatches = check_weight_checksums(checksum_dir=tmp_path)
        assert mismatches == []

    def test_param_hash_mismatch_reports_correct_details(self, tmp_path: Path) -> None:
        entry_ok = _make_entry(param_hashes={"pp0.weight": "aaa"})
        entry_bad = _make_entry(param_hashes={"pp0.weight": "zzz"})

        _write_entry(tmp_path, step=5, rank=0, entry=entry_ok)
        _write_entry(tmp_path, step=5, rank=1, entry=entry_bad)
        _write_entry(tmp_path, step=5, rank=2, entry=entry_ok)

        mismatches = check_weight_checksums(checksum_dir=tmp_path)

        assert len(mismatches) == 1
        m = mismatches[0]
        assert m.step == 5
        assert m.tensor_category == "param"
        assert m.tensor_name == "pp0.weight"
        assert 0 in m.cell_indices
        assert 1 in m.cell_indices

    def test_missing_tensor_in_one_replica_reports_mismatch(self, tmp_path: Path) -> None:
        entry_full = _make_entry(param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"})
        entry_partial = _make_entry(param_hashes={"pp0.weight": "aaa"})

        _write_entry(tmp_path, step=0, rank=0, entry=entry_full)
        _write_entry(tmp_path, step=0, rank=1, entry=entry_partial)

        mismatches = check_weight_checksums(checksum_dir=tmp_path)

        assert len(mismatches) == 1
        m = mismatches[0]
        assert m.tensor_name == "pp0.bias"
        assert "<missing>" in m.hashes

    def test_multiple_steps_only_reports_mismatched_step(self, tmp_path: Path) -> None:
        good_entry = _make_entry(param_hashes={"pp0.weight": "aaa"})
        bad_entry = _make_entry(param_hashes={"pp0.weight": "zzz"})

        # Step 0: all match
        _write_entry(tmp_path, step=0, rank=0, entry=good_entry)
        _write_entry(tmp_path, step=0, rank=1, entry=good_entry)

        # Step 1: mismatch
        _write_entry(tmp_path, step=1, rank=0, entry=good_entry)
        _write_entry(tmp_path, step=1, rank=1, entry=bad_entry)

        # Step 2: all match
        _write_entry(tmp_path, step=2, rank=0, entry=good_entry)
        _write_entry(tmp_path, step=2, rank=1, entry=good_entry)

        mismatches = check_weight_checksums(checksum_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].step == 1

    def test_empty_directory_returns_no_mismatches(self, tmp_path: Path) -> None:
        mismatches = check_weight_checksums(checksum_dir=tmp_path)
        assert mismatches == []

    def test_buffer_mismatch_detected(self, tmp_path: Path) -> None:
        entry_a = _make_entry(buffer_hashes={"pp0.running_mean": "aaa"})
        entry_b = _make_entry(buffer_hashes={"pp0.running_mean": "bbb"})

        _write_entry(tmp_path, step=0, rank=0, entry=entry_a)
        _write_entry(tmp_path, step=0, rank=1, entry=entry_b)

        mismatches = check_weight_checksums(checksum_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].tensor_category == "buffer"

    def test_optimizer_state_mismatch_detected(self, tmp_path: Path) -> None:
        entry_a = _make_entry(optimizer_state_hashes={"pp0.weight/exp_avg": "aaa"})
        entry_b = _make_entry(optimizer_state_hashes={"pp0.weight/exp_avg": "bbb"})

        _write_entry(tmp_path, step=3, rank=0, entry=entry_a)
        _write_entry(tmp_path, step=3, rank=1, entry=entry_b)

        mismatches = check_weight_checksums(checksum_dir=tmp_path)

        assert len(mismatches) == 1
        assert mismatches[0].tensor_category == "optimizer_state"
        assert mismatches[0].tensor_name == "pp0.weight/exp_avg"
