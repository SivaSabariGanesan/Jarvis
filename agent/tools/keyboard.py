"""
Controlled Keyboard input tools for JARVIS V4.
Provides text typing via Unicode SendInput, key presses, and validated hotkeys.
"""

import sys
import time
import ctypes
from typing import List, Optional
from ctypes import wintypes

from agent.tools.security import (
    security_validator,
    RiskLevel,
    SecurityViolation,
    ConfirmationRequired,
    SecurityAuditLogger,
)

# Win32 Virtual-Key Map
VK_MAP = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "insert": 0x2D,
    "delete": 0x2E,
    "del": 0x2E,
    "win": 0x5B,
    "windows": 0x5B,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}

# Win32 Flags
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


def _send_unicode_char(char: str):
    """Send a single character as Unicode input to avoid layout mismatches."""
    if sys.platform != "win32":
        return

    char_code = ord(char)

    # Key Down
    inp_down = INPUT()
    inp_down.type = INPUT_KEYBOARD
    inp_down.union.ki = KEYBDINPUT(wVk=0, wScan=char_code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=0)

    # Key Up
    inp_up = INPUT()
    inp_up.type = INPUT_KEYBOARD
    inp_up.union.ki = KEYBDINPUT(wVk=0, wScan=char_code, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)

    inputs = (INPUT * 2)(inp_down, inp_up)
    ctypes.windll.user32.SendInput(2, inputs, ctypes.sizeof(INPUT))


def _get_vk_code(key: str) -> int:
    """Get the Win32 virtual-key code for a given key string."""
    k = key.lower().strip()
    if k in VK_MAP:
        return VK_MAP[k]
    if len(k) == 1:
        if "a" <= k <= "z":
            return ord(k.upper())
        if "0" <= k <= "9":
            return ord(k)
    raise SecurityViolation(f"Unrecognized key code for '{key}'.")


def _send_key_down(vk_code: int):
    if sys.platform == "win32":
        ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)


def _send_key_up(vk_code: int):
    if sys.platform == "win32":
        ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)


def type_text(text: str) -> str:
    """
    Type text as keyboard input into the currently focused application.
    Does NOT execute commands; only generates standard keystrokes.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    if not isinstance(text, str):
        raise SecurityViolation("Text must be a valid string.")

    if len(text) > 1000:
        raise SecurityViolation("Text exceeds maximum allowed length of 1000 characters.")

    for char in text:
        _send_unicode_char(char)
        time.sleep(0.01)

    summary = f"Typed text ({len(text)} characters)."
    SecurityAuditLogger.log_execution("type_text", True, summary)
    return summary


def press_key(key: str) -> str:
    """
    Press a single key from the authorized keyboard allowlist (e.g. 'enter', 'tab', 'esc', 'backspace').
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    validated_key = security_validator.validate_key(key)
    vk = _get_vk_code(validated_key)

    _send_key_down(vk)
    time.sleep(0.05)
    _send_key_up(vk)

    summary = f"Pressed key '{validated_key}'."
    SecurityAuditLogger.log_execution("press_key", True, summary)
    return summary


def hotkey(keys: str) -> str:
    """
    Execute a keyboard shortcut combination (e.g. 'ctrl+c', 'ctrl+v', 'alt+tab').
    High-risk combinations (such as 'shift+delete' or 'alt+f4') require explicit confirmation.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    validated_keys, risk_level = security_validator.validate_hotkey(keys)

    # Press modifiers down
    vk_codes = [_get_vk_code(k) for k in validated_keys]
    try:
        for vk in vk_codes:
            _send_key_down(vk)
            time.sleep(0.03)
        time.sleep(0.05)
    finally:
        # Release in reverse order
        for vk in reversed(vk_codes):
            _send_key_up(vk)
            time.sleep(0.02)

    combo_str = "+".join(validated_keys)
    summary = f"Executed hotkey '{combo_str}'."
    SecurityAuditLogger.log_execution("hotkey", True, summary)
    return summary
