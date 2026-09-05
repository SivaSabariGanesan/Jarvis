"""
Controlled Mouse input tools for JARVIS V4.
Provides secure cursor positioning, clicking, and scrolling with screen boundary validation.
"""

import sys
import time
import ctypes
from typing import Tuple, Optional
from ctypes import wintypes

from agent.tools.security import security_validator, SecurityViolation, SecurityAuditLogger

# Win32 Mouse Event Flags
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120


def get_screen_dimensions() -> Tuple[int, int]:
    """Get active primary monitor width and height."""
    if sys.platform == "win32":
        w = ctypes.windll.user32.GetSystemMetrics(0)
        h = ctypes.windll.user32.GetSystemMetrics(1)
        return int(w), int(h)
    return 1920, 1080


def get_cursor_position() -> Tuple[int, int]:
    """Get current mouse cursor coordinates."""
    if sys.platform == "win32":
        pt = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)
    return 0, 0


def move_mouse(x: int, y: int) -> str:
    """
    Move the mouse cursor to specific coordinates on screen.
    Coordinates must be within the actual screen dimensions.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    w, h = get_screen_dimensions()
    target_x, target_y = security_validator.validate_mouse_coordinates(x, y, w, h)

    if sys.platform == "win32":
        ctypes.windll.user32.SetCursorPos(target_x, target_y)

    summary = f"Moved mouse to ({target_x}, {target_y})."
    SecurityAuditLogger.log_execution("move_mouse", True, summary)
    return summary


def click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    Click the primary (left) mouse button, optionally moving to (x, y) first.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    if x is not None and y is not None:
        move_mouse(x, y)

    if sys.platform == "win32":
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.05)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    cur_x, cur_y = get_cursor_position()
    summary = f"Clicked at ({cur_x}, {cur_y})."
    SecurityAuditLogger.log_execution("click", True, summary)
    return summary


def double_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    Double click the left mouse button at current position or specified coordinates.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    if x is not None and y is not None:
        move_mouse(x, y)

    if sys.platform == "win32":
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.08)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    cur_x, cur_y = get_cursor_position()
    summary = f"Double-clicked at ({cur_x}, {cur_y})."
    SecurityAuditLogger.log_execution("double_click", True, summary)
    return summary


def right_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    Right click the mouse at current position or specified coordinates.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    if x is not None and y is not None:
        move_mouse(x, y)

    if sys.platform == "win32":
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        time.sleep(0.05)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    cur_x, cur_y = get_cursor_position()
    summary = f"Right-clicked at ({cur_x}, {cur_y})."
    SecurityAuditLogger.log_execution("right_click", True, summary)
    return summary


def scroll(amount: int) -> str:
    """
    Scroll the mouse wheel vertically. Positive values scroll up, negative scroll down.
    Amount represents the number of click notches (1 to 20).
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    if not isinstance(amount, int) or isinstance(amount, bool):
        raise SecurityViolation("Scroll amount must be an integer.")

    clamped_amount = max(-20, min(20, amount))
    wheel_delta = clamped_amount * WHEEL_DELTA

    if sys.platform == "win32":
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_delta, 0)

    direction = "up" if clamped_amount > 0 else "down"
    summary = f"Scrolled {direction} by {abs(clamped_amount)} steps."
    SecurityAuditLogger.log_execution("scroll", True, summary)
    return summary
