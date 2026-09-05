"""
Controlled Browser Tools for JARVIS V5.
Provides safe browser launching, URL navigation, search, history navigation,
and state inspection without arbitrary shell or DOM access.
"""

import sys
import time
import urllib.parse
import webbrowser
import logging
from typing import Dict, Any, Optional

from agent.config import settings
from agent.tools.security import (
    security_validator,
    RiskLevel,
    SecurityViolation,
    ConfirmationRequired,
    SecurityAuditLogger,
)
from agent.tools.applications import open_application, is_application_running
from agent.tools.window_info import get_active_window, get_open_windows
from agent.tools.keyboard import type_text, press_key, hotkey
from agent.tools.action_queue import wait_for_window

logger = logging.getLogger("jarvis.tools.browser")

ALLOWED_BROWSERS = {"chrome", "edge", "firefox", "brave"}
BROWSER_PROCESS_NAMES = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"}


def get_browser_state() -> Dict[str, Any]:
    """
    Inspect whether an authorized browser is currently open and active on the desktop.
    Returns structured window telemetry without accessing private browser cookies, tokens, or passwords.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    active = get_active_window()
    active_proc = str(active.get("process", "")).lower()
    is_browser_active = active_proc in BROWSER_PROCESS_NAMES

    open_wins = get_open_windows()
    running_browser = None
    for win in open_wins:
        proc = str(win.get("process", "")).lower()
        if proc in BROWSER_PROCESS_NAMES:
            running_browser = win
            break

    browser_name = "None"
    if is_browser_active:
        browser_name = active_proc.replace(".exe", "").capitalize()
    elif running_browser:
        browser_name = str(running_browser.get("process", "")).replace(".exe", "").capitalize()

    state = {
        "browser": browser_name,
        "running": running_browser is not None or is_browser_active,
        "active": is_browser_active,
        "window_title": active.get("title") if is_browser_active else (running_browser.get("title") if running_browser else "None"),
        "bounds": active.get("bounds") if is_browser_active else (running_browser.get("bounds") if running_browser else None),
    }
    SecurityAuditLogger.log_execution("get_browser_state", True, f"Browser: {browser_name}, Running: {state['running']}")
    return state


def open_browser(browser_name: str = "chrome") -> str:
    """
    Launch or focus an authorized browser (e.g. 'chrome' or 'edge') using the application allowlist.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    clean_name = browser_name.strip().lower()
    if clean_name not in ALLOWED_BROWSERS:
        raise SecurityViolation(
            f"Browser '{browser_name}' is not in the authorized browser list ({', '.join(sorted(ALLOWED_BROWSERS))})."
        )

    # Use deterministic application tool
    result = open_application(clean_name)
    SecurityAuditLogger.log_execution("open_browser", True, f"Launched {clean_name}")
    return f"{result}"


def open_url(url: str) -> str:
    """
    Open an authorized http/https URL in the default browser.
    Rejects dangerous schemes (file://, javascript:, data:, etc.) and validates domain policies.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    validated_url = security_validator.validate_browser_url(url)

    try:
        # Safe OS/browser dispatch without shell=True
        webbrowser.open(validated_url)
        summary = f"Navigated to '{validated_url}'."
        SecurityAuditLogger.log_execution("open_url", True, summary)
        return summary
    except Exception as e:
        SecurityAuditLogger.log_execution("open_url", False, str(e))
        return f"Failed to open URL: {e}"


async def browser_search(query: str) -> str:
    """
    Execute a web search for the specified query string.
    Treats search query strictly as data, never as executable code.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    if not query or not str(query).strip():
        raise SecurityViolation("Search query cannot be empty.")

    clean_query = str(query).strip()
    encoded_query = urllib.parse.quote_plus(clean_query)
    search_url = f"https://www.google.com/search?q={encoded_query}"

    # Validate generated URL
    validated_url = security_validator.validate_browser_url(search_url)

    # Dispatch navigation
    webbrowser.open(validated_url)
    time.sleep(0.5)

    summary = f"Searched the web for '{clean_query}'."
    SecurityAuditLogger.log_execution("browser_search", True, summary)
    return summary


def browser_go_back() -> str:
    """
    Navigate back in the active browser history.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    hotkey("alt+left")
    summary = "Navigated back in browser history."
    SecurityAuditLogger.log_execution("browser_go_back", True, summary)
    return summary


def browser_go_forward() -> str:
    """
    Navigate forward in the active browser history.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    hotkey("alt+right")
    summary = "Navigated forward in browser history."
    SecurityAuditLogger.log_execution("browser_go_forward", True, summary)
    return summary


def browser_refresh() -> str:
    """
    Refresh the currently active browser page.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    press_key("f5")
    summary = "Refreshed current browser page."
    SecurityAuditLogger.log_execution("browser_refresh", True, summary)
    return summary
