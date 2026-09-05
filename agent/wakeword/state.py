"""
State Machine for JARVIS V2 Voice Agent.
Enforces explicit state transitions: IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE.
"""

import enum
import logging
import threading
from typing import Callable, Optional, Set

logger = logging.getLogger("jarvis.wakeword.state")


class AgentState(str, enum.Enum):
    """Explicit states for JARVIS Voice Agent."""
    IDLE = "IDLE"              # Waiting for wake word "Jarvis"
    LISTENING = "LISTENING"    # Capturing user speech command after wake word
    PROCESSING = "PROCESSING"  # STT transcription, LLM reasoning, or TTS generation
    SPEAKING = "SPEAKING"      # Playing audio response through speakers


# Valid state transitions
VALID_TRANSITIONS = {
    AgentState.IDLE: {AgentState.LISTENING, AgentState.PROCESSING, AgentState.SPEAKING},
    AgentState.LISTENING: {AgentState.PROCESSING, AgentState.IDLE},
    AgentState.PROCESSING: {AgentState.SPEAKING, AgentState.IDLE},
    AgentState.SPEAKING: {AgentState.IDLE, AgentState.LISTENING},
}


class StateMachine:
    """
    Thread-safe Finite State Machine managing JARVIS operational state.
    """

    def __init__(self, initial_state: AgentState = AgentState.IDLE):
        self._state = initial_state
        self._lock = threading.RLock()
        self._listeners: list[Callable[[AgentState, AgentState], None]] = []

    @property
    def current_state(self) -> AgentState:
        with self._lock:
            return self._state

    def is_state(self, state: AgentState) -> bool:
        with self._lock:
            return self._state == state

    def add_listener(self, listener: Callable[[AgentState, AgentState], None]) -> None:
        """Register a callback for state transition events: callback(old_state, new_state)."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[AgentState, AgentState], None]) -> None:
        """Unregister a state transition callback."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def transition_to(self, new_state: AgentState, force: bool = False) -> bool:
        """
        Transition to a new state if valid.
        Returns True if transition succeeded, False otherwise.
        """
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return True

            valid_targets = VALID_TRANSITIONS.get(old_state, set())
            if not force and new_state not in valid_targets:
                logger.warning(
                    f"Invalid state transition attempted: {old_state.value} -> {new_state.value}. Ignored."
                )
                return False

            self._state = new_state
            logger.info(f"[STATE] Transition: {old_state.value} -> {new_state.value}")

            # Notify listeners
            listeners_copy = list(self._listeners)

        for listener in listeners_copy:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state transition listener: {e}")

        return True

    def reset_to_idle(self) -> None:
        """Reset state machine unconditionally to IDLE."""
        self.transition_to(AgentState.IDLE, force=True)
