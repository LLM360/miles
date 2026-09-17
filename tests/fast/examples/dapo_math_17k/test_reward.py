import asyncio
import multiprocessing
import time
from types import SimpleNamespace

from examples.dapo_math_17k import reward
from sympy import Integer, Sum, binomial, symbols


def _never_returns() -> float:
    while True:
        time.sleep(1)


def test_isolated_process_returns_result():
    assert reward._run_in_isolated_process(float, (3,), timeout_seconds=5) == 3.0


def test_isolated_process_timeout_terminates_child():
    initial_child_pids = {process.pid for process in multiprocessing.active_children()}
    started_at = time.monotonic()
    result = reward._run_in_isolated_process(_never_returns, (), timeout_seconds=0.1)

    assert result == 0.0
    assert time.monotonic() - started_at < 5
    assert {process.pid for process in multiprocessing.active_children()} == initial_child_pids


def test_oversized_finite_sum_skips_verify(monkeypatch):
    index = symbols("i", integer=True)
    prediction = Sum((-1) ** index * binomial(2010, index), (index, 0, 2010))

    monkeypatch.setattr(reward, "_parse_gold", lambda _: [Integer(2)])
    monkeypatch.setattr(reward, "parse", lambda *args, **kwargs: [prediction])

    def fail_verify(*args, **kwargs):
        raise AssertionError("verify must not run for an oversized finite sum")

    monkeypatch.setattr(reward, "verify", fail_verify)

    assert reward._score_response("2", "oversized symbolic sum") == 0.0


def test_small_finite_sum_still_uses_verify(monkeypatch):
    index = symbols("i", integer=True)
    prediction = Sum(index, (index, 0, 10))

    monkeypatch.setattr(reward, "_parse_gold", lambda _: [Integer(55)])
    monkeypatch.setattr(reward, "parse", lambda *args, **kwargs: [prediction])
    monkeypatch.setattr(reward, "verify", lambda *args, **kwargs: True)

    assert reward._score_response("55", "small symbolic sum") == 1.0


def test_reward_func_preserves_sample_order(monkeypatch):
    async def fake_score(sample):
        await asyncio.sleep(0)
        return float(sample.label)

    monkeypatch.setattr(reward, "_score_sample", fake_score)
    samples = [SimpleNamespace(label="1"), SimpleNamespace(label="0")]

    assert asyncio.run(reward.reward_func(None, samples)) == [1.0, 0.0]


def test_reward_func_scores_boxed_answers_in_subprocesses():
    samples = [
        SimpleNamespace(label="42", response=r"Answer: \boxed{42}"),
        SimpleNamespace(label="42", response=r"Answer: \boxed{41}"),
    ]

    assert asyncio.run(reward.reward_func(None, samples)) == [1.0, 0.0]
