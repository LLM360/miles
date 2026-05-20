import os
import time

import pytest
from tests.fast.dashboard.dummy_dump import dump_dummy_run

from miles.dashboard.dump_reader import DumpReader, DumpStillWriting
from miles.utils.rollout_dump import load_rollout_dump, save_rollout_dump


def test_parquet_discovery_join_summary_and_lazy_token_mirror(tmp_path):
    truth = dump_dummy_run(tmp_path, steps=2, dp_size=2, with_eval=True, duplicate_first_sample_index=True)
    old_reader = DumpReader(tmp_path)
    expected = old_reader.summary(0)
    for path in (tmp_path / "rollout_data").glob("*.pt"):
        save_rollout_dump(path.with_suffix(".parquet"), load_rollout_dump(path))
        os.utime(path.with_suffix(".parquet"), (time.time() - 100,) * 2)
        path.unlink()
    for path in (tmp_path / "dashboard_columns").glob("rollout_*.parquet"):
        path.unlink()
    reader = DumpReader(tmp_path, cache_dir=tmp_path / "new_cache")
    assert reader.rollout_ids().train == [0, 1]
    assert reader.rollout_ids().eval == truth.eval_ids
    joined = reader.load_joined(0)
    assert joined.train_coverage == 1.0
    assert reader.summary(0).equals(expected)
    assert reader.load_joined(0, evaluation=True).samples
    for occurrence in [0, 1]:
        columns = reader._rollout_columns(0, joined.samples[0].index, sample_occurrence=occurrence, evaluation=False)
        assert columns["tokens"] == joined.samples[occurrence].tokens
    assert (tmp_path / "dashboard_columns" / "rollout_0.parquet").exists()


def test_dual_format_newest_dump_wins_and_ids_are_deduplicated(tmp_path):
    directory = tmp_path / "rollout_data"
    directory.mkdir()
    for suffix, stamp in [("pt", 100), ("parquet", 200)]:
        path = directory / f"0.{suffix}"
        save_rollout_dump(path, dict(rollout_id=0, samples=[], metadata={"format": suffix}))
        os.utime(path, (stamp, stamp))
    (directory / "unrelated.parquet").touch()
    reader = DumpReader(tmp_path)
    assert reader.rollout_ids().train == [0]
    assert "0.parquet" in reader._source_stamps(0, evaluation=False)
    os.utime(directory / "0.pt", (300, 300))
    assert "0.pt" in reader._source_stamps(0, evaluation=False)


def test_fresh_broken_parquet_uses_existing_retry_behavior(tmp_path):
    directory = tmp_path / "rollout_data"
    directory.mkdir()
    (directory / "0.parquet").write_bytes(b"incomplete")
    with pytest.raises(DumpStillWriting):
        DumpReader(tmp_path).load_joined(0)
