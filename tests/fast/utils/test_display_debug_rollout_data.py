import pytest
import typer
from typer.testing import CliRunner

from miles.utils.debug_utils.display_debug_rollout_data import main
from miles.utils.rollout_dump import save_rollout_dump
from miles.utils.types import Sample


@pytest.fixture
def cli():
    app = typer.Typer()
    app.command()(main)
    return lambda args: CliRunner().invoke(app, args)


def _record(tmp_path, suffix, *, empty=False):
    sample = Sample(index=3, tokens=[1, 2, 3], response_length=2, reward=1.0)
    save_rollout_dump(
        tmp_path / f"4.{suffix}",
        dict(rollout_id=4, samples=[] if empty else [sample.to_dict()], metadata={}),
    )
    return ["--load-debug-rollout-data", str(tmp_path / "{rollout_id}.pt"), "--category", "train"]


@pytest.mark.parametrize("suffix", ["pt", "parquet"])
@pytest.mark.parametrize("timing", [[], ["--rollout-time", "2"], ["--rollout-time", "2", "--rollout-num-gpus", "2"]])
def test_cli_reads_both_formats_with_optional_speed_metrics(cli, tmp_path, suffix, timing):
    result = cli(_record(tmp_path, suffix) + timing)
    assert result.exit_code == 0, result.output
    assert '"index": 3' in result.output
    if not timing:
        assert "Speed metrics unavailable" in result.output
        assert "tokens_per_sec" not in result.output
    else:
        assert "'longest_sample_tokens_per_sec': 1.0" in result.output
        if "--rollout-num-gpus" in timing:
            assert "'tokens_per_gpu_per_sec': 0.5" in result.output
        else:
            assert "tokens_per_gpu_per_sec" not in result.output


@pytest.mark.parametrize(
    "option,value", [("--rollout-time", "0"), ("--rollout-time", "-1"), ("--rollout-num-gpus", "0")]
)
def test_cli_rejects_nonpositive_timing_inputs(cli, tmp_path, option, value):
    result = cli(_record(tmp_path, "pt") + [option, value])
    assert result.exit_code == 2
    assert "must be positive" in result.output


def test_empty_dump_does_not_try_to_compute_speed(cli, tmp_path):
    result = cli(_record(tmp_path, "parquet", empty=True) + ["--rollout-time", "2"])
    assert result.exit_code == 0, result.output
    assert "dump contains no samples" in result.output


def test_metrics_can_be_disabled(cli, tmp_path):
    result = cli(_record(tmp_path, "parquet") + ["--no-show-metrics"])
    assert result.exit_code == 0, result.output
    assert '"index": 3' in result.output
    assert "Speed metrics unavailable" not in result.output


def test_help_exposes_timing_options(cli):
    result = cli(["--help"])
    assert result.exit_code == 0, result.output
    assert "rollout-time" in result.output and "rollout-num-gpus" in result.output
