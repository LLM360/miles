from types import SimpleNamespace

from miles.rollout.inference_rollout.inference_rollout_train import (
    _abort_signal_budgets,
    _resolve_rollout_abort_timeout,
)


def test_default_abort_timeout_preserves_nested_contract():
    total_timeout = _resolve_rollout_abort_timeout(SimpleNamespace())
    signal_timeout, harbor_timeout = _abort_signal_budgets(total_timeout)

    assert total_timeout == 180.0
    assert harbor_timeout == 126.0
    assert signal_timeout == 130.0
    assert 120.0 < 122.0 < harbor_timeout < signal_timeout < total_timeout


def test_explicit_abort_timeout_is_preserved():
    args = SimpleNamespace(rollout_abort_timeout_seconds=45)

    assert _resolve_rollout_abort_timeout(args) == 45.0


def test_abort_budgets_remain_bounded_by_short_overall_deadline():
    signal_timeout, harbor_timeout = _abort_signal_budgets(8.0)

    assert signal_timeout == 8.0
    assert harbor_timeout == 4.0
    assert harbor_timeout < signal_timeout
