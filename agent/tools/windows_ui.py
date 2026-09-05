"""
Windows UI and navigation tools for JARVIS V3.
Enforces secure execution without arbitrary shells.
"""

import os
import sys
import ctypes
import logging
import webbrowser
from pathlib import Path
from typing import Optional

from agent.tools.security import security_validator, SecurityAuditLogger, SecurityViolation

logger = logging.getLogger("jarvis.tools.windows_ui")


def open_start_menu() -> str:
    """Open the Windows Start menu by simulating the Windows key."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        VK_LWIN = 0x5B
        KEYEVENTF_KEYUP = 0x0002
        user32.keybd_event(VK_LWIN, 0, 0, 0)
        user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
        SecurityAuditLogger.log_execution("open_start_menu", True, "Triggered Start menu")
        return "Start menu opened, sir."
    return "Start menu is only supported on Windows."


def open_settings(page: Optional[str] = None) -> str:
    """
    Open Windows Settings.
    Optionally accepts a specific known settings page (e.g. sound, display, network, bluetooth).
    """
    settings_pages = {
        "sound": "ms-settings:sound",
        "display": "ms-settings:display",
        "network": "ms-settings:network",
        "wifi": "ms-settings:network-wifi",
        "bluetooth": "ms-settings:bluetooth",
        "apps": "ms-settings:appsfeatures",
        "update": "ms-settings:windowsupdate",
        "battery": "ms-settings:batterysaver",
        "storage": "ms-settings:storagesense",
    }

    target_uri = "ms-settings:"
    if page:
        page_clean = page.strip().lower()
        if page_clean in settings_pages:
            target_uri = settings_pages[page_clean]

    if sys.platform == "win32":
        try:
            os.startfile(target_uri)
            SecurityAuditLogger.log_execution("open_settings", True, f"Opened {target_uri}")
            return f"Windows Settings{' for ' + page if page else ''} is open, sir."
        except Exception as e:
            SecurityAuditLogger.log_execution("open_settings", False, f"Failed: {e}")
            return f"Could not open Windows Settings: {e}"
    return "Windows Settings is only supported on Windows."


def open_task_manager() -> str:
    """Open Windows Task Manager."""
    if sys.platform == "win32":
        try:
            os.startfile("taskmgr.exe")
            SecurityAuditLogger.log_execution("open_task_manager", True, "Opened Task Manager")
            return "Task Manager is open, sir."
        except Exception as e:
            SecurityAuditLogger.log_execution("open_task_manager", False, f"Failed: {e}")
            return f"Could not open Task Manager: {e}"
    return "Task Manager is only supported on Windows."


def open_file_explorer(path: Optional[str] = None) -> str:
    """
    Open Windows File Explorer.
    If path is provided, validates that it is within the allowed workspace.
    """
    target_path = ""
    if path:
        validated = security_validator.validate_path(path)
        target_path = str(validated)

    if sys.platform == "win32":
        try:
            if target_path:
                os.startfile(target_path)
            else:
                os.startfile("explorer.exe")
            SecurityAuditLogger.log_execution("open_file_explorer", True, f"Opened explorer at '{target_path or 'default'}'")
            return f"File Explorer opened{' at ' + target_path if target_path else ''}, sir."
        except Exception as e:
            SecurityAuditLogger.log_execution("open_file_explorer", False, f"Failed: {e}")
            return f"Could not open File Explorer: {e}"
    return "File Explorer is only supported on Windows."


def open_browser_url(url: str) -> str:
    """
    Open a web URL in the user's default browser.
    Validates that URL uses HTTP/HTTPS scheme and rejects file:// or javascript: schemes.
    """
    clean_url = security_validator.validate_url(url)
    try:
        webbrowser.open(clean_url)
        SecurityAuditLogger.log_execution("open_browser_url", True, f"Opened URL: {clean_url}")
        return f"Opened {clean_url} in your default browser, sir."
    except Exception as e:
        SecurityAuditLogger.log_execution("open_browser_url", False, f"Failed to open URL: {e}")
        return f"Could not open browser URL: {e}"
