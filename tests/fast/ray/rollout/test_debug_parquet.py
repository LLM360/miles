import json
from pathlib import Path

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout import debug_data
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.utils.rollout_dump import load_rollout_dump


def _args(tmp_path, **kwargs):
    template = str(tmp_path / "rollout_data" / "r_[test]_{rollout_id}.pt")
    return make_args(
        save_debug_rollout_data=template,
        load_debug_rollout_data=template,
        ci_inject_rollout_data_path=template,
        save_rollout_format="parquet",
        save_debug_trajectory_data=str(tmp_path / "trajectory" / "{rollout_id}.jsonl"),
        **kwargs,
    )


def test_parquet_replay_injection_and_sidecars(tmp_path):
    args = _args(tmp_path)
    messages = [{"role": "user", "content": "hello"}]
    sample = make_sample(index=9, metadata={"messages": messages})
    save_debug_rollout_data(args, [sample], 7, evaluation=False, metadata={"dynamic_global_batch_size": 4})
    path = tmp_path / "rollout_data" / "r_[test]_7.parquet"
    assert path.exists() and not path.with_suffix(".pt").exists()
    assert (tmp_path / "dashboard_columns" / "rollout_7.parquet").exists()
    assert json.loads((tmp_path / "trajectory" / "7.jsonl").read_text())["messages"] == messages
    for samples, metadata in (load_debug_rollout_data(args, 7), RolloutDataInjectionUtil.load(args, 7)):
        assert samples[0].index == 9
        assert "messages" not in samples[0].metadata
        assert metadata == {"dynamic_global_batch_size": 4}


@pytest.mark.parametrize("retain", [0, 1, 2, 4])
def test_retention_handles_gaps_formats_and_preserves_unrelated_files(tmp_path, retain):
    args = _args(tmp_path, save_rollout_retain_last_n=0)
    for rollout_id, format in [(2, "pt"), (5, "parquet"), (9, "pt")]:
        args.save_rollout_format = format
        save_debug_rollout_data(args, [make_sample(metadata={"messages": []})], rollout_id, evaluation=False)
    save_debug_rollout_data(args, {"eval": {"samples": [make_sample(metadata={"messages": []})]}}, 2, evaluation=True)
    foreign = [
        tmp_path / "rollout_data" / "foreign_1.pt",
        tmp_path / "train_data" / "2_0.pt",
        tmp_path / "checkpoint" / "2.pt",
        tmp_path / "trajectory" / "foreign_2.jsonl",
    ]
    for path in foreign:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep")
    args.save_rollout_retain_last_n = retain
    args.save_rollout_format = "parquet"
    save_debug_rollout_data(args, [make_sample(metadata={"messages": []})], 10, evaluation=False)
    for rollout_id in [2, 5, 9, 10]:
        retained = retain == 0 or rollout_id > 10 - retain
        template = Path(args.save_debug_rollout_data.format(rollout_id=rollout_id))
        assert (template.exists() or template.with_suffix(".parquet").exists()) == retained
        assert (tmp_path / "dashboard_columns" / f"rollout_{rollout_id}.parquet").exists() == retained
        assert (tmp_path / "trajectory" / f"{rollout_id}.jsonl").exists() == retained
    assert (tmp_path / "rollout_data" / "r_[test]_eval_2.pt").exists()
    assert (tmp_path / "trajectory" / "eval_2.jsonl").exists()
    assert (tmp_path / "dashboard_columns" / "rollout_eval_2.parquet").exists()
    assert all(path.read_text() == "keep" for path in foreign)


def test_eval_save_and_failed_train_save_do_not_prune(tmp_path, monkeypatch):
    args = _args(tmp_path, save_rollout_retain_last_n=1)
    save_debug_rollout_data(args, [make_sample()], 0, evaluation=False)
    old_path = Path(args.save_debug_rollout_data.format(rollout_id=0)).with_suffix(".parquet")
    save_debug_rollout_data(args, {"eval": {"samples": [make_sample()]}}, 10, evaluation=True)
    assert old_path.exists()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(debug_data, "save_rollout_dump", fail)
    with pytest.raises(OSError, match="disk full"):
        save_debug_rollout_data(args, [make_sample()], 10, evaluation=False)
    assert load_rollout_dump(old_path)["rollout_id"] == 0
