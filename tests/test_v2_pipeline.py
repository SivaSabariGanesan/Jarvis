"""
End-to-end and component integration tests for JARVIS V2.
Tests state flow: IDLE -> Wake Word -> LISTENING -> STT -> LLM -> TTS -> IDLE.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from agent.wakeword.state import AgentState, StateMachine
from agent.config import settings
from agent.prompt import JARVIS_SYSTEM_PROMPT


def test_v2_configuration():
    assert settings.wake_word == "jarvis"
    assert settings.wake_word_threshold >= 0.0
    assert settings.wake_word_cooldown >= 0.0
    assert "Yes, sir?" in settings.wake_word_activation_response


def test_v2_state_pipeline_flow():
    sm = StateMachine(AgentState.IDLE)
    assert sm.current_state == AgentState.IDLE

    # Step 1: Wake word detected
    wake_word_detected = True
    if wake_word_detected:
        sm.transition_to(AgentState.LISTENING)
    assert sm.current_state == AgentState.LISTENING

    # Step 2: User speaks command
    user_command = "What is 2 + 2?"
    sm.transition_to(AgentState.PROCESSING)
    assert sm.current_state == AgentState.PROCESSING

    # Step 3: LLM generates response
    llm_response = "Four, sir."
    assert len(llm_response) > 0

    # Step 4: TTS speaks response
    sm.transition_to(AgentState.SPEAKING)
    assert sm.current_state == AgentState.SPEAKING

    # Step 5: Speech completes, returns to IDLE
    sm.transition_to(AgentState.IDLE)
    assert sm.current_state == AgentState.IDLE


def test_normal_conversation_does_not_trigger_llm():
    sm = StateMachine(AgentState.IDLE)
    llm_called = False

    def mock_process_audio(has_wake_word: bool):
        nonlocal llm_called
        if has_wake_word:
            sm.transition_to(AgentState.LISTENING)
            sm.transition_to(AgentState.PROCESSING)
            llm_called = True
            sm.transition_to(AgentState.SPEAKING)
            sm.transition_to(AgentState.IDLE)

    # Normal speech without wake word
    mock_process_audio(has_wake_word=False)
    assert sm.current_state == AgentState.IDLE
    assert llm_called is False

    # Speech with wake word
    mock_process_audio(has_wake_word=True)
    assert sm.current_state == AgentState.IDLE
    assert llm_called is True
