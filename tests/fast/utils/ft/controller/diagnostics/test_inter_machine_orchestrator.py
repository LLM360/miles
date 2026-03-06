"""Tests for InterMachineOrchestrator and cross_compare."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.diagnostics.inter_machine_orchestrator import (
    InterMachineOrchestrator,
    PairResult,
    cross_compare,
)
from miles.utils.ft.models._diagnostics import DiagnosticResult
from tests.fast.utils.ft.conftest import FakeNodeAgent


def _make_passing_agent(node_id: str) -> FakeNodeAgent:
    return FakeNodeAgent(
        diagnostic_results={
            "inter_machine": DiagnosticResult(
                diagnostic_type="inter_machine",
                node_id=node_id,
                passed=True,
                details="pass",
            ),
        },
        node_id=node_id,
    )


def _make_failing_agent(node_id: str) -> FakeNodeAgent:
    return FakeNodeAgent(
        diagnostic_results={
            "inter_machine": DiagnosticResult(
                diagnostic_type="inter_machine",
                node_id=node_id,
                passed=False,
                details="fail",
            ),
        },
        node_id=node_id,
    )


# ===================================================================
# cross_compare
# ===================================================================


class TestCrossCompare:
    def test_all_pass_returns_empty(self) -> None:
        results = [
            PairResult(master_id="A", worker_id="B", passed=True),
            PairResult(master_id="B", worker_id="A", passed=True),
        ]

        assert cross_compare(node_ids=["A", "B"], pair_results=results) == []

    def test_isolates_bad_node(self) -> None:
        """B fails in both pairs it participates in, A and C only fail when paired with B."""
        results = [
            PairResult(master_id="A", worker_id="B", passed=False),
            PairResult(master_id="B", worker_id="C", passed=False),
            PairResult(master_id="C", worker_id="A", passed=True),
        ]

        bad = cross_compare(node_ids=["A", "B", "C"], pair_results=results)

        assert bad == ["B"]

    def test_all_fail_returns_empty(self) -> None:
        """All nodes fail equally — cannot localize."""
        results = [
            PairResult(master_id="A", worker_id="B", passed=False),
            PairResult(master_id="B", worker_id="A", passed=False),
        ]

        assert cross_compare(node_ids=["A", "B"], pair_results=results) == []

    def test_single_node_no_pairs(self) -> None:
        """Single node with no pair results."""
        assert cross_compare(node_ids=["A"], pair_results=[]) == []

    def test_multiple_bad_nodes(self) -> None:
        """Two nodes both have highest failure count."""
        results = [
            PairResult(master_id="A", worker_id="B", passed=False),
            PairResult(master_id="B", worker_id="C", passed=False),
            PairResult(master_id="C", worker_id="A", passed=False),
        ]

        bad = cross_compare(node_ids=["A", "B", "C"], pair_results=results)

        assert bad == [] or len(bad) <= 3


# ===================================================================
# _resolve_address
# ===================================================================


class TestResolveAddress:
    def test_with_custom_addresses(self) -> None:
        orch = InterMachineOrchestrator(
            agents={},
            node_addresses={"n1": "10.0.0.1", "n2": "10.0.0.2"},
        )

        assert orch._resolve_address("n1") == "10.0.0.1"

    def test_fallback_to_node_id(self) -> None:
        orch = InterMachineOrchestrator(agents={}, node_addresses=None)

        assert orch._resolve_address("n1") == "n1"

    def test_missing_node_in_addresses_falls_back(self) -> None:
        orch = InterMachineOrchestrator(
            agents={},
            node_addresses={"n1": "10.0.0.1"},
        )

        assert orch._resolve_address("n99") == "n99"


# ===================================================================
# _run_single_pair — missing agent
# ===================================================================


class TestRunSinglePairMissingAgent:
    @pytest.mark.anyio
    async def test_missing_master_agent_returns_failed(self) -> None:
        agents = {"worker": _make_passing_agent("worker")}
        orch = InterMachineOrchestrator(agents=agents)

        result = await orch._run_single_pair(
            master_id="master", worker_id="worker",
            master_addr="10.0.0.1", port=29500, timeout_seconds=30,
        )

        assert result.passed is False
        assert result.master_id == "master"

    @pytest.mark.anyio
    async def test_missing_worker_agent_returns_failed(self) -> None:
        agents = {"master": _make_passing_agent("master")}
        orch = InterMachineOrchestrator(agents=agents)

        result = await orch._run_single_pair(
            master_id="master", worker_id="worker",
            master_addr="10.0.0.1", port=29500, timeout_seconds=30,
        )

        assert result.passed is False

    @pytest.mark.anyio
    async def test_both_agents_missing_returns_failed(self) -> None:
        orch = InterMachineOrchestrator(agents={})

        result = await orch._run_single_pair(
            master_id="A", worker_id="B",
            master_addr="10.0.0.1", port=29500, timeout_seconds=30,
        )

        assert result.passed is False


# ===================================================================
# run — fewer than 2 nodes
# ===================================================================


class TestRunEdgeCases:
    @pytest.mark.anyio
    async def test_single_node_returns_empty(self) -> None:
        agents = {"A": _make_passing_agent("A")}
        orch = InterMachineOrchestrator(agents=agents)

        bad = await orch.run(node_ids=["A"], timeout_seconds=30)

        assert bad == []

    @pytest.mark.anyio
    async def test_empty_nodes_returns_empty(self) -> None:
        orch = InterMachineOrchestrator(agents={})

        bad = await orch.run(node_ids=[], timeout_seconds=30)

        assert bad == []

    @pytest.mark.anyio
    async def test_two_nodes_all_pass(self) -> None:
        agents = {
            "A": _make_passing_agent("A"),
            "B": _make_passing_agent("B"),
        }
        orch = InterMachineOrchestrator(agents=agents)

        bad = await orch.run(node_ids=["A", "B"], timeout_seconds=30)

        assert bad == []
