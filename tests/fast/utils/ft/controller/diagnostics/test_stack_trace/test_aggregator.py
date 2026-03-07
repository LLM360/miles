"""Tests for StackTraceAggregator."""

from __future__ import annotations

from miles.utils.ft.agents.diagnostics.stack_trace import PySpyFrame, PySpyThread
from miles.utils.ft.controller.diagnostics.stack_trace import StackTraceAggregator
from tests.fast.utils.ft.helpers import (
    SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
    SAMPLE_PYSPY_THREADS_NORMAL,
    SAMPLE_PYSPY_THREADS_STUCK,
)


class TestStackTraceAggregatorBasic:
    def test_empty_traces_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        assert agg.aggregate(traces={}) == []

    def test_single_node_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        result = agg.aggregate(traces={"node-0": SAMPLE_PYSPY_THREADS_NORMAL})
        assert result == []

    def test_all_same_traces_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-1": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-2": SAMPLE_PYSPY_THREADS_NORMAL,
        }
        result = agg.aggregate(traces=traces)
        assert result == []


class TestStackTraceAggregatorSuspectDetection:
    def test_one_different_node_is_suspect(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_STUCK,
            "node-1": SAMPLE_PYSPY_THREADS_STUCK,
            "node-2": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert result == ["node-2"]

    def test_two_nodes_all_different_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_STUCK,
            "node-1": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert result == []

    def test_multiple_minority_groups(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-1": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-2": SAMPLE_PYSPY_THREADS_NORMAL,
            "node-3": SAMPLE_PYSPY_THREADS_STUCK,
            "node-4": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert "node-3" in result
        assert "node-4" in result
        assert "node-0" not in result

    def test_tied_groups_returns_empty(self) -> None:
        agg = StackTraceAggregator()
        traces = {
            "node-0": SAMPLE_PYSPY_THREADS_STUCK,
            "node-1": SAMPLE_PYSPY_THREADS_STUCK,
            "node-2": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
            "node-3": SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK,
        }
        result = agg.aggregate(traces=traces)
        assert result == []


class TestStackTraceAggregatorFingerprint:
    def test_fingerprint_ignores_line_numbers(self) -> None:
        """Same function+filename but different line numbers produce same fingerprint."""
        agg = StackTraceAggregator()
        threads_a = [
            PySpyThread(
                thread_name="MainThread", active=True, owns_gil=False,
                frames=[
                    PySpyFrame(name="func_a", filename="file.py", line=10),
                    PySpyFrame(name="func_b", filename="file.py", line=20),
                ],
            ),
        ]
        threads_b = [
            PySpyThread(
                thread_name="MainThread", active=True, owns_gil=False,
                frames=[
                    PySpyFrame(name="func_a", filename="file.py", line=99),
                    PySpyFrame(name="func_b", filename="file.py", line=88),
                ],
            ),
        ]
        assert agg._extract_fingerprint(threads_a) == agg._extract_fingerprint(threads_b)

    def test_fingerprint_differs_on_function_name(self) -> None:
        agg = StackTraceAggregator()
        threads_a = [
            PySpyThread(
                thread_name="MainThread", active=True, owns_gil=False,
                frames=[PySpyFrame(name="func_a", filename="file.py", line=10)],
            ),
        ]
        threads_b = [
            PySpyThread(
                thread_name="MainThread", active=True, owns_gil=False,
                frames=[PySpyFrame(name="func_DIFFERENT", filename="file.py", line=10)],
            ),
        ]
        assert agg._extract_fingerprint(threads_a) != agg._extract_fingerprint(threads_b)

    def test_fingerprint_uses_innermost_frame_only(self) -> None:
        agg = StackTraceAggregator()
        threads = [
            PySpyThread(
                thread_name="MainThread", active=True, owns_gil=False,
                frames=[
                    PySpyFrame(name="innermost_func", filename="inner.py", line=1),
                    PySpyFrame(name="middle_func", filename="mid.py", line=2),
                    PySpyFrame(name="outermost_func", filename="outer.py", line=3),
                ],
            ),
        ]
        fp = agg._extract_fingerprint(threads)
        assert "innermost_func" in fp
        assert "outermost_func" not in fp

    def test_fingerprint_skips_empty_frame_threads(self) -> None:
        agg = StackTraceAggregator()
        threads = [
            PySpyThread(
                thread_name="EmptyThread", active=False, owns_gil=False,
                frames=[],
            ),
        ]
        assert agg._extract_fingerprint(threads) == ""

    def test_real_sample_produces_nonempty_fingerprint(self) -> None:
        agg = StackTraceAggregator()
        fp = agg._extract_fingerprint(SAMPLE_PYSPY_THREADS_NORMAL)
        assert fp != ""
        assert "(" in fp
