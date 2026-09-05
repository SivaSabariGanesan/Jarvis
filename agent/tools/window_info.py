"""
Safe window inspection tools for JARVIS V4.
Provides active window details and open window enumeration without exposing arbitrary process manipulation.
"""

import sys
import ctypes
from typing import Dict, List, Any, Optional
from ctypes import wintypes
import psutil

from agent.tools.security import SecurityAuditLogger


def _get_process_name_from_hwnd(hwnd: int) -> str:
    """Safely get process name for a given window handle."""
    try:
        pid = wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value > 0:
            proc = psutil.Process(pid.value)
            return proc.name()
    except Exception:
        pass
    return "unknown"


def _get_window_rect(hwnd: int) -> Optional[List[int]]:
    """Get [x, y, width, height] of a window."""
    try:
        rect = wintypes.RECT()
        if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            return [int(rect.left), int(rect.top), int(w), int(h)]
    except Exception:
        pass
    return None


def get_active_window() -> Dict[str, Any]:
    """
    Get information about the currently focused/active desktop window.
    Returns window title, process name, and screen bounding box.
    """
    if sys.platform != "win32":
        return {"title": "Desktop", "process": "explorer.exe", "bounds": [0, 0, 1920, 1080]}

    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        return {"title": "None", "process": "unknown", "bounds": None}

    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    buff = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
    title = buff.value.strip()

    proc_name = _get_process_name_from_hwnd(hwnd)
    rect = _get_window_rect(hwnd)

    info = {
        "title": title or "(Untitled Window)",
        "process": proc_name,
        "bounds": rect,
        "is_active": True,
    }
    SecurityAuditLogger.log_execution("get_active_window", True, f"Active: {info['title']} ({proc_name})")
    return info


def get_open_windows() -> List[Dict[str, Any]]:
    """
    List user-facing desktop application windows currently open and visible.
    """
    if sys.platform != "win32":
        return [{"title": "Main Window", "process": "explorer.exe", "bounds": [0, 0, 1920, 1080]}]

    results: List[Dict[str, Any]] = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def enum_windows_callback(hwnd, lparam):
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True

        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buff = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value.strip()

        # Skip shell artifacts, ToolTips, Program Manager
        if not title or title in ("Program Manager", "Settings", "Default IME", "MSCTFIME UI"):
            return True

        rect = _get_window_rect(hwnd)
        if not rect or rect[2] < 100 or rect[3] < 100:
            return True

        proc_name = _get_process_name_from_hwnd(hwnd)
        results.append({
            "title": title,
            "process": proc_name,
            "bounds": rect,
        })
        return True

    cb = WNDENUMPROC(enum_windows_callback)
    ctypes.windll.user32.EnumWindows(cb, 0)

    SecurityAuditLogger.log_execution("get_open_windows", True, f"Found {len(results)} open windows.")
    return results
