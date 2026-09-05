"""
JARVIS Wake-Word Subsystem.
Provides local offline wake-word detection and finite state machine management.
"""

from agent.wakeword.state import AgentState, StateMachine
from agent.wakeword.detector import BaseWakeWordDetector, OpenWakeWordDetector

__all__ = [
    "AgentState",
    "StateMachine",
    "BaseWakeWordDetector",
    "OpenWakeWordDetector",
]
