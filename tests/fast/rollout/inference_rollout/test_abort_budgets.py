from miles.rollout.inference_rollout.inference_rollout_train import (
    _abort_signal_budgets,
)


def test_agentic_abort_deadlines_are_strictly_nested():
    signal_timeout, harbor_timeout = _abort_signal_budgets(120.0)

    assert harbor_timeout == 66.0
    assert signal_timeout == 70.0
    assert 60.0 < 62.0 < harbor_timeout < signal_timeout < 120.0


def test_abort_budgets_remain_bounded_by_short_overall_deadline():
    signal_timeout, harbor_timeout = _abort_signal_budgets(8.0)

    assert signal_timeout == 8.0
    assert harbor_timeout == 4.0
    assert harbor_timeout < signal_timeout
