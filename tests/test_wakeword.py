"""
Unit tests for Local Wake Word Detector.
Verifies detector initialization, model loading, threshold evaluation,
cooldown enforcement, and self-trigger suppression.
"""

import time
import pytest
import numpy as np
from agent.wakeword.detector import OpenWakeWordDetector


def test_detector_initialization():
    detector = OpenWakeWordDetector(wake_word="jarvis", threshold=0.5, cooldown_seconds=1.0)
    assert detector.wake_word == "jarvis"
    assert detector.threshold == 0.5
    assert detector.cooldown_seconds == 1.0
    assert detector._model is not None
    assert detector._model_key is not None


def test_detector_silent_frame():
    detector = OpenWakeWordDetector(wake_word="jarvis", threshold=0.5)
    silent_frame = np.zeros(1280, dtype=np.int16)

    # Silent frame should not trigger detection
    detected = detector.process_frame(silent_frame)
    assert detected is False


def test_detector_suppression_self_trigger_protection():
    detector = OpenWakeWordDetector(wake_word="jarvis", threshold=0.5)
    detector.set_suppressed(True)
    assert detector.is_suppressed is True

    # When suppressed, processing frames always returns False
    random_frame = np.random.randint(-1000, 1000, 1280, dtype=np.int16)
    assert detector.process_frame(random_frame) is False

    detector.set_suppressed(False)
    assert detector.is_suppressed is False


def test_detector_cooldown():
    detector = OpenWakeWordDetector(wake_word="jarvis", threshold=0.0, cooldown_seconds=2.0)
    detector._last_detection_time = time.time() - 0.5  # Only 0.5s ago

    dummy_frame = np.zeros(1280, dtype=np.int16)
    # Should be rejected because within 2.0s cooldown
    assert detector.process_frame(dummy_frame) is False

    # Simulate time passed beyond cooldown
    detector._last_detection_time = time.time() - 3.0
    # Now cooldown check passes (prediction executes)
    detector.reset()
