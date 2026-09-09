from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    P2PTransferManager,
)


def _completed_future() -> Future:
    future = Future()
    future.set_result(None)
    return future


def test_wait_transfers_drains_successful_tasks_and_clears_queue() -> None:
    manager = P2PTransferManager(transfer_timeout=7.0)
    first = MagicMock(wraps=_completed_future())
    second = MagicMock(wraps=_completed_future())
    manager.transfer_futures = [first, second]

    manager.wait_transfers()

    first.result.assert_called_once_with(timeout=7.0)
    second.result.assert_called_once_with(timeout=7.0)
    assert manager.transfer_futures == []


def test_wait_transfers_drains_all_tasks_then_raises_aggregate_error() -> None:
    manager = P2PTransferManager(transfer_timeout=3.0)
    first = MagicMock()
    second = MagicMock()
    third = MagicMock()
    first.result.side_effect = RuntimeError("session-a failed")
    second.result.return_value = None
    third.result.side_effect = ValueError("session-c failed")
    manager.transfer_futures = [first, second, third]

    with pytest.raises(RuntimeError, match=r"2 of 3 P2P weight transfers failed") as exc_info:
        manager.wait_transfers()

    first.result.assert_called_once_with(timeout=3.0)
    second.result.assert_called_once_with(timeout=3.0)
    third.result.assert_called_once_with(timeout=3.0)
    assert "session-a failed" in str(exc_info.value)
    assert "session-c failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert manager.transfer_futures == []


def test_wait_transfers_propagates_timeout_and_clears_queue() -> None:
    manager = P2PTransferManager(transfer_timeout=0.25)
    timed_out = MagicMock()
    completed = MagicMock()
    timed_out.result.side_effect = TimeoutError("transfer timed out")
    completed.result.return_value = None
    manager.transfer_futures = [timed_out, completed]

    with pytest.raises(RuntimeError, match=r"1 of 2 P2P weight transfers failed") as exc_info:
        manager.wait_transfers()

    timed_out.result.assert_called_once_with(timeout=0.25)
    completed.result.assert_called_once_with(timeout=0.25)
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert manager.transfer_futures == []
