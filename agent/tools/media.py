"""
Media playback and volume controls for JARVIS V3.
Uses standard Windows user32 keybd_event without external process execution.
"""

import sys
import ctypes
import logging
from agent.tools.security import SecurityAuditLogger

logger = logging.getLogger("jarvis.tools.media")

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002


def _send_key(vk_code: int):
    """Simulate a multimedia key event on Windows."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
        user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


def media_play_pause() -> str:
    """Toggle play/pause for active media players."""
    _send_key(VK_MEDIA_PLAY_PAUSE)
    SecurityAuditLogger.log_execution("media_play_pause", True, "Toggled play/pause")
    return "Media playback toggled, sir."


def volume_up(steps: int = 2) -> str:
    """Increase system volume."""
    safe_steps = max(1, min(int(steps), 10))
    for _ in range(safe_steps):
        _send_key(VK_VOLUME_UP)
    SecurityAuditLogger.log_execution("volume_up", True, f"Volume increased by {safe_steps} steps")
    return "Volume turned up, sir."


def volume_down(steps: int = 2) -> str:
    """Decrease system volume."""
    safe_steps = max(1, min(int(steps), 10))
    for _ in range(safe_steps):
        _send_key(VK_VOLUME_DOWN)
    SecurityAuditLogger.log_execution("volume_down", True, f"Volume decreased by {safe_steps} steps")
    return "Volume turned down, sir."


def volume_mute() -> str:
    """Toggle volume mute."""
    _send_key(VK_VOLUME_MUTE)
    SecurityAuditLogger.log_execution("volume_mute", True, "Toggled mute")
    return "Audio mute toggled, sir."
