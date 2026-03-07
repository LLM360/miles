"""Tests for StateMachineStepper and StateMachine base classes."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from miles.utils.ft.utils.state_machine import StateMachine, StateMachineStepper


# -- Dummy states for testing --------------------------------------------------


class DummyState(BaseModel):
    model_config = ConfigDict(frozen=True)


class StateA(DummyState):
    pass


class StateB(DummyState):
    value: int = 0


class StateC(DummyState):
    pass


class TerminalState(DummyState):
    pass


class UnregisteredState(DummyState):
    pass


# -- Dummy handlers ------------------------------------------------------------


class StateAHandler:
    async def step(self, state: StateA, _context: None) -> DummyState:
        return StateB(value=1)


class StateBHandler:
    async def step(self, state: StateB, _context: None) -> DummyState | None:
        if state.value >= 3:
            return TerminalState()
        return StateB(value=state.value + 1)


class StateCHandler:
    async def step(self, state: StateC, _context: None) -> DummyState | None:
        return None


class TerminalStateHandler:
    async def step(self, state: TerminalState, _context: None) -> DummyState | None:
        return None


HANDLER_MAP: dict[type, type] = {
    StateA: StateAHandler,
    StateB: StateBHandler,
    StateC: StateCHandler,
    TerminalState: TerminalStateHandler,
}


def _make_stepper(**kwargs) -> StateMachineStepper[DummyState, None]:
    return StateMachineStepper(handler_map=HANDLER_MAP, **kwargs)


# -- Tests: StateMachineStepper ------------------------------------------------


class TestStateMachineStepper:
    @pytest.mark.asyncio
    async def test_dispatch_to_correct_handler(self) -> None:
        stepper = _make_stepper()
        result = await stepper(StateA(), None)
        assert isinstance(result, StateB)
        assert result.value == 1

    @pytest.mark.asyncio
    async def test_terminal_state_returns_none(self) -> None:
        stepper = _make_stepper()
        result = await stepper(TerminalState(), None)
        assert result is None

    @pytest.mark.asyncio
    async def test_handler_returning_none(self) -> None:
        stepper = _make_stepper()
        result = await stepper(StateC(), None)
        assert result is None

    @pytest.mark.asyncio
    async def test_unregistered_state_raises_type_error(self) -> None:
        stepper = _make_stepper()
        with pytest.raises(TypeError, match="has no handler for state type UnregisteredState"):
            await stepper(UnregisteredState(), None)

    @pytest.mark.asyncio
    async def test_same_type_transition(self) -> None:
        stepper = _make_stepper()
        result = await stepper(StateB(value=1), None)
        assert isinstance(result, StateB)
        assert result.value == 2

    @pytest.mark.asyncio
    async def test_transition_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        stepper = _make_stepper()
        with caplog.at_level("INFO"):
            await stepper(StateA(), None)
        assert "StateA()" in caplog.text
        assert "StateB(value=1)" in caplog.text

    @pytest.mark.asyncio
    async def test_same_type_transition_logs_when_data_changes(self, caplog: pytest.LogCaptureFixture) -> None:
        """StateB(value=1) -> StateB(value=2) should still be logged."""
        stepper = _make_stepper()
        with caplog.at_level("INFO"):
            await stepper(StateB(value=1), None)
        assert "StateB(value=1)" in caplog.text
        assert "StateB(value=2)" in caplog.text

    @pytest.mark.asyncio
    async def test_same_state_no_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Handler returning None (no transition) should not log."""
        stepper = _make_stepper()
        with caplog.at_level("INFO"):
            await stepper(StateC(), None)
        assert caplog.text == ""

    @pytest.mark.asyncio
    async def test_pre_dispatch_short_circuits(self) -> None:
        """pre_dispatch returning non-None skips handler dispatch."""

        async def always_terminal(state: DummyState, ctx: None) -> DummyState | None:
            return TerminalState()

        stepper = _make_stepper(pre_dispatch=always_terminal)
        result = await stepper(StateA(), None)
        assert isinstance(result, TerminalState)

    @pytest.mark.asyncio
    async def test_pre_dispatch_none_falls_through(self) -> None:
        """pre_dispatch returning None continues to normal handler dispatch."""

        async def pass_through(state: DummyState, ctx: None) -> DummyState | None:
            return None

        stepper = _make_stepper(pre_dispatch=pass_through)
        result = await stepper(StateA(), None)
        assert isinstance(result, StateB)
        assert result.value == 1


# -- Tests: StateMachine -------------------------------------------------------


class TestStateMachine:
    @pytest.mark.asyncio
    async def test_step_runs_until_none(self) -> None:
        """StateA -> StateB(1) -> StateB(2) -> StateB(3) -> TerminalState -> (stepper returns None)."""
        machine = StateMachine(initial_state=StateA(), stepper=_make_stepper())
        await machine.step(None)

        assert isinstance(machine.state, TerminalState)
        assert len(machine.state_history) == 4

    @pytest.mark.asyncio
    async def test_step_no_transition(self) -> None:
        machine = StateMachine(initial_state=StateC(), stepper=_make_stepper())
        await machine.step(None)
        assert isinstance(machine.state, StateC)
        assert len(machine.state_history) == 0

    @pytest.mark.asyncio
    async def test_step_already_terminal(self) -> None:
        machine = StateMachine(initial_state=TerminalState(), stepper=_make_stepper())
        await machine.step(None)
        assert isinstance(machine.state, TerminalState)
        assert len(machine.state_history) == 0

    @pytest.mark.asyncio
    async def test_state_history_records_all_transitions(self) -> None:
        machine = StateMachine(initial_state=StateA(), stepper=_make_stepper())
        await machine.step(None)

        types = [type(s).__name__ for s in machine.state_history]
        assert types == ["StateB", "StateB", "StateB", "TerminalState"]

    @pytest.mark.asyncio
    async def test_stepper_property(self) -> None:
        stepper = _make_stepper()
        machine = StateMachine(initial_state=StateA(), stepper=stepper)
        assert machine.stepper is stepper
