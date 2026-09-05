"""
Unit and security tests for JARVIS V4 Input Tools (Mouse, Keyboard, Window Info).
"""

import math
import pytest
from agent.tools.security import (
    security_validator,
    RiskLevel,
    SecurityViolation,
    ConfirmationRequired,
)
from agent.tools.mouse import (
    get_screen_dimensions,
    move_mouse,
    click,
    double_click,
    right_click,
    scroll,
)
from agent.tools.keyboard import type_text, press_key, hotkey
from agent.tools.window_info import get_active_window, get_open_windows


@pytest.fixture(autouse=True)
def ensure_security_resumed():
    """Ensure computer control is active before and after tests."""
    security_validator.resume_control()
    yield
    security_validator.resume_control()


def test_mouse_coordinate_validation_valid():
    """Test valid coordinates within active screen resolution."""
    w, h = get_screen_dimensions()
    assert w > 0 and h > 0

    # Test center coordinates
    cx, cy = w // 2, h // 2
    res_x, res_y = security_validator.validate_mouse_coordinates(cx, cy, w, h)
    assert res_x == cx
    assert res_y == cy

    # Test origin (0, 0) and max bounds (w-1, h-1)
    assert security_validator.validate_mouse_coordinates(0, 0, w, h) == (0, 0)
    assert security_validator.validate_mouse_coordinates(w - 1, h - 1, w, h) == (w - 1, h - 1)


@pytest.mark.parametrize("invalid_x, invalid_y", [
    (-1, 500),
    (500, -1),
    (-100, -100),
    (99999, 500),
    (500, 99999),
    (float("nan"), 500),
    (500, float("nan")),
    (float("inf"), 500),
    (500, float("inf")),
    ("invalid", 500),
    (True, 500),
    (500, False),
])
def test_mouse_coordinate_validation_rejected(invalid_x, invalid_y):
    """Test out-of-bounds, negative, NaN, Inf, and malformed coordinates are rejected."""
    w, h = 1920, 1080
    with pytest.raises(SecurityViolation):
        security_validator.validate_mouse_coordinates(invalid_x, invalid_y, w, h)


def test_mouse_actions_under_emergency_stop():
    """Ensure all mouse actions are strictly blocked when emergency stop is active."""
    security_validator.pause_control(reason="Test Stop")

    with pytest.raises(SecurityViolation, match="paused"):
        move_mouse(100, 100)

    with pytest.raises(SecurityViolation, match="paused"):
        click(100, 100)

    with pytest.raises(SecurityViolation, match="paused"):
        double_click(100, 100)

    with pytest.raises(SecurityViolation, match="paused"):
        right_click(100, 100)

    with pytest.raises(SecurityViolation, match="paused"):
        scroll(5)


def test_keyboard_type_text_pure_data():
    """Ensure type_text accepts standard text and treats command strings purely as keystrokes."""
    result = type_text("hello && calc")
    assert "Typed text" in result

    # Reject non-string or oversized inputs
    with pytest.raises(SecurityViolation):
        type_text(12345)  # type: ignore

    with pytest.raises(SecurityViolation, match="maximum allowed length"):
        type_text("A" * 1500)


@pytest.mark.parametrize("valid_key", [
    "enter", "tab", "esc", "space", "backspace", "delete", "up", "down", "left", "right", "f5"
])
def test_keyboard_press_key_allowlist_accepted(valid_key):
    """Ensure allowlisted keys are validated and accepted."""
    assert security_validator.validate_key(valid_key) == valid_key.lower()


@pytest.mark.parametrize("invalid_key", [
    "malicious_key", "system_execute", "power_off", "", " "
])
def test_keyboard_press_key_allowlist_rejected(invalid_key):
    """Ensure unauthorized keys are rejected with SecurityViolation."""
    with pytest.raises(SecurityViolation):
        security_validator.validate_key(invalid_key)


def test_safe_hotkeys_classification():
    """Safe hotkeys should classify as LOW risk."""
    for hk in ["ctrl+c", "ctrl+v", "ctrl+z", "alt+tab", "win+d"]:
        keys, risk = security_validator.validate_hotkey(hk)
        assert risk == RiskLevel.LOW
        assert len(keys) >= 2


def test_dangerous_hotkeys_classification():
    """Dangerous / destructive hotkeys should classify as HIGH risk."""
    for hk in ["shift+delete", "alt+f4", "ctrl+alt+delete"]:
        keys, risk = security_validator.validate_hotkey(hk)
        assert risk == RiskLevel.HIGH


def test_keyboard_actions_under_emergency_stop():
    """Ensure all keyboard actions are blocked when emergency stop is active."""
    security_validator.pause_control(reason="Test Stop")

    with pytest.raises(SecurityViolation, match="paused"):
        type_text("test")

    with pytest.raises(SecurityViolation, match="paused"):
        press_key("enter")

    with pytest.raises(SecurityViolation, match="paused"):
        hotkey("ctrl+c")


def test_window_info_tools():
    """Ensure get_active_window and get_open_windows return structured data."""
    active = get_active_window()
    assert isinstance(active, dict)
    assert "title" in active
    assert "process" in active

    open_wins = get_open_windows()
    assert isinstance(open_wins, list)
    for win in open_wins:
        assert "title" in win
        assert "process" in win
        assert "bounds" in win
