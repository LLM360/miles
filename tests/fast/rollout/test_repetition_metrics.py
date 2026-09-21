from copy import deepcopy
from types import SimpleNamespace

import pytest

from miles.ray.rollout import metrics as rollout
from miles.utils.repetition import record_repetition
from miles.utils.types import Sample


@pytest.fixture
def args():
    return SimpleNamespace(
        reward_key=None,
        advantage_estimator="ppo",
        sglang_speculative_algorithm=None,
        log_reward_category=None,
        log_problem_category="domain",
        ci_test=False,
        custom_rollout_log_function_path=None,
        custom_eval_rollout_log_function_path=None,
        load_debug_rollout_data=None,
        log_passrate=False,
        wandb_always_use_train_step=False,
        rollout_batch_size=3,
        n_samples_per_prompt=1,
        global_batch_size=3,
        rollout_num_gpus=0,
    )


@pytest.fixture
async def samples():
    repeated = Sample(
        tokens=[1] * 10001,
        response="",
        response_length=10001,
        reward=1.0,
        metadata={"response_decoded": False, "domain": "fmp"},
        status=Sample.Status.COMPLETED,
    )
    await record_repetition(repeated, SimpleNamespace(decode=lambda tokens: "x" * len(tokens)))
    return [
        repeated,
        Sample(
            response="short", response_length=5, reward=0.0, metadata={"domain": "fmp"}, status=Sample.Status.COMPLETED
        ),
        Sample(
            tokens=[2],
            response="",
            response_length=1,
            reward=0.0,
            metadata={"response_decoded": False, "domain": "unknown"},
            status=Sample.Status.TRUNCATED,
        ),
    ]


def assert_repetition_metrics(metrics):
    for prefix, fraction, coverage in [
        ("", 0.5, 2 / 3),
        ("response_stats/", 0.5, 2 / 3),
        ("response_stats/correct/", 1.0, 1.0),
        ("response_stats/incorrect/", 0.0, 0.5),
        ("response_stats/fmp/", 0.5, 1.0),
        ("response_stats/fmp/correct/", 1.0, 1.0),
        ("response_stats/fmp/incorrect/", 0.0, 1.0),
    ]:
        assert metrics[f"{prefix}repetition_frac"] == fraction
        assert metrics[f"{prefix}repetition_coverage"] == coverage
    for prefix in ["response_stats/unknown/", "response_stats/unknown/incorrect/"]:
        assert metrics[f"{prefix}repetition_coverage"] == 0.0
        assert f"{prefix}repetition_frac" not in metrics


def test_overall_and_all_group_splits_use_known_denominator_once(args, samples, monkeypatch):
    before = deepcopy([s.to_dict() for s in samples])
    calls = []
    original = rollout.sample_repetition

    def spy(sample):
        calls.append(id(sample))
        return original(sample)

    monkeypatch.setattr(rollout, "sample_repetition", spy)
    metrics = rollout._compute_metrics_from_samples(args, samples)

    assert_repetition_metrics(metrics)
    assert calls == [id(s) for s in samples]
    assert metrics["reward/raw_reward"] == pytest.approx(1 / 3)
    assert metrics["reward/correctness"] == pytest.approx(1 / 3)
    assert metrics["truncated_ratio"] == pytest.approx(1 / 3)
    assert [s.to_dict() for s in samples] == before


@pytest.mark.parametrize("evaluation", [False, True])
def test_train_and_eval_logging_expose_coverage(args, samples, monkeypatch, evaluation):
    captured = {}
    monkeypatch.setattr(rollout.tracking, "log", lambda _args, metrics, step_key: captured.update(metrics))
    if evaluation:
        rollout.log_eval_rollout_data(3, args, {"heldout": {"samples": samples, "rewards": [1.0, 0.0, 0.0]}})
        prefix = "eval/heldout/"
        assert captured["eval/heldout"] == pytest.approx(1 / 3)
    else:
        rollout.log_rollout_data(3, args, samples, {}, 1.0)
        prefix = "rollout/"
        assert captured["response_stats/repetition_coverage"] == 2 / 3
    assert_repetition_metrics(
        {key.removeprefix(prefix): value for key, value in captured.items() if key.startswith(prefix)}
    )


def test_all_unknown_omits_every_repetition_fraction(args, samples):
    metrics = rollout._compute_metrics_from_samples(args, samples[2:])
    coverage = {k: v for k, v in metrics.items() if k.endswith("repetition_coverage")}
    assert coverage
    assert set(coverage.values()) == {0.0}
    assert not any(k.endswith("repetition_frac") for k in metrics)


def test_all_legacy_text_preserves_previous_repetition_values(args):
    samples = [Sample(response="x" * 10001, reward=1.0), Sample(response="short", reward=0.0)]
    metrics = rollout._compute_metrics_from_samples(args, samples)
    assert metrics["repetition_frac"] == 0.5
    assert metrics["repetition_coverage"] == 1.0
    assert metrics["response_stats/correct/repetition_frac"] == 1.0
    assert metrics["response_stats/incorrect/repetition_frac"] == 0.0
