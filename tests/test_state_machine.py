"""
Unit tests for JARVIS V2 State Machine.
Verifies state transitions, invalid transition rejections, and listener callbacks.
"""

import pytest
from agent.wakeword.state import AgentState, StateMachine, VALID_TRANSITIONS


def test_initial_state():
    sm = StateMachine()
    assert sm.current_state == AgentState.IDLE
    assert sm.is_state(AgentState.IDLE)
    assert not sm.is_state(AgentState.LISTENING)


def test_valid_transitions():
    sm = StateMachine()
    # IDLE -> LISTENING
    assert sm.transition_to(AgentState.LISTENING) is True
    assert sm.current_state == AgentState.LISTENING

    # LISTENING -> PROCESSING
    assert sm.transition_to(AgentState.PROCESSING) is True
    assert sm.current_state == AgentState.PROCESSING

    # PROCESSING -> SPEAKING
    assert sm.transition_to(AgentState.SPEAKING) is True
    assert sm.current_state == AgentState.SPEAKING

    # SPEAKING -> IDLE
    assert sm.transition_to(AgentState.IDLE) is True
    assert sm.current_state == AgentState.IDLE


def test_invalid_transitions_rejected():
    sm = StateMachine(AgentState.IDLE)
    # Transition to LISTENING
    sm.transition_to(AgentState.LISTENING)
    # LISTENING cannot go directly to SPEAKING
    assert sm.transition_to(AgentState.SPEAKING) is False
    assert sm.current_state == AgentState.LISTENING

    # Transition to PROCESSING
    sm.transition_to(AgentState.PROCESSING)
    # PROCESSING cannot go directly to LISTENING
    assert sm.transition_to(AgentState.LISTENING) is False
    assert sm.current_state == AgentState.PROCESSING


def test_listener_callback():
    sm = StateMachine()
    recorded_events = []

    def on_change(old_state, new_state):
        recorded_events.append((old_state, new_state))

    sm.add_listener(on_change)
    sm.transition_to(AgentState.LISTENING)
    sm.transition_to(AgentState.PROCESSING)

    assert len(recorded_events) == 2
    assert recorded_events[0] == (AgentState.IDLE, AgentState.LISTENING)
    assert recorded_events[1] == (AgentState.LISTENING, AgentState.PROCESSING)

    # Test listener removal
    sm.remove_listener(on_change)
    sm.transition_to(AgentState.SPEAKING)
    assert len(recorded_events) == 2


def test_reset_to_idle():
    sm = StateMachine(AgentState.PROCESSING)
    assert sm.current_state == AgentState.PROCESSING
    sm.reset_to_idle()
    assert sm.current_state == AgentState.IDLE
