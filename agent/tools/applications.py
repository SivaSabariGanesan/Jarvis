"""
Application launcher and process manager tool for JARVIS V3.
Enforces a strict executable allowlist and denies arbitrary executable execution.
"""

import os
import sys
import logging
import subprocess
from typing import Dict, Any, Optional
import psutil

from agent.tools.security import RiskLevel, SecurityViolation, SecurityAuditLogger

logger = logging.getLogger("jarvis.tools.applications")

# Strict Application Allowlist (Keys are lowercase normalized lookup names)
APPLICATIONS_ALLOWLIST: Dict[str, Dict[str, Any]] = {
    "word": {
        "display_name": "Microsoft Word",
        "executable": "WINWORD.EXE",
        "command": "winword",
        "risk": RiskLevel.LOW,
    },
    "excel": {
        "display_name": "Microsoft Excel",
        "executable": "EXCEL.EXE",
        "command": "excel",
        "risk": RiskLevel.LOW,
    },
    "powerpoint": {
        "display_name": "Microsoft PowerPoint",
        "executable": "POWERPNT.EXE",
        "command": "powerpnt",
        "risk": RiskLevel.LOW,
    },
    "chrome": {
        "display_name": "Google Chrome",
        "executable": "chrome.exe",
        "command": "chrome",
        "risk": RiskLevel.LOW,
    },
    "edge": {
        "display_name": "Microsoft Edge",
        "executable": "msedge.exe",
        "command": "msedge",
        "risk": RiskLevel.LOW,
    },
    "notepad": {
        "display_name": "Notepad",
        "executable": "notepad.exe",
        "command": "notepad",
        "risk": RiskLevel.LOW,
    },
    "calculator": {
        "display_name": "Calculator",
        "executable": "calc.exe",
        "command": "calc",
        "risk": RiskLevel.LOW,
    },
    "vscode": {
        "display_name": "Visual Studio Code",
        "executable": "Code.exe",
        "command": "code",
        "risk": RiskLevel.LOW,
    },
    "paint": {
        "display_name": "Paint",
        "executable": "mspaint.exe",
        "command": "mspaint",
        "risk": RiskLevel.LOW,
    },
    "spotify": {
        "display_name": "Spotify",
        "executable": "Spotify.exe",
        "command": "spotify",
        "risk": RiskLevel.LOW,
    },
    "explorer": {
        "display_name": "File Explorer",
        "executable": "explorer.exe",
        "command": "explorer",
        "risk": RiskLevel.LOW,
    },
    "taskmanager": {
        "display_name": "Task Manager",
        "executable": "Taskmgr.exe",
        "command": "taskmgr",
        "risk": RiskLevel.LOW,
    },
}


def normalize_app_name(name: str) -> Optional[str]:
    """Resolve user-spoken application name to allowlist key."""
    if not name:
        return None
    cleaned = name.strip().lower()

    # Direct match
    if cleaned in APPLICATIONS_ALLOWLIST:
        return cleaned

    # Alias matching
    aliases = {
        "microsoft word": "word",
        "ms word": "word",
        "doc": "word",
        "google chrome": "chrome",
        "browser": "chrome",
        "calc": "calculator",
        "vs code": "vscode",
        "visual studio code": "vscode",
        "code": "vscode",
        "ms edge": "edge",
        "text editor": "notepad",
        "file explorer": "explorer",
        "files": "explorer",
        "task manager": "taskmanager",
        "taskmgr": "taskmanager",
        "ms paint": "paint",
    }
    if cleaned in aliases:
        return aliases[cleaned]

    # Substring search in display names
    for key, info in APPLICATIONS_ALLOWLIST.items():
        if cleaned in key or cleaned in info["display_name"].lower():
            return key

    return None


def open_application(application: str) -> str:
    """
    Open an authorized application from the strict allowlist.
    Never executes arbitrary commands or shells.
    """
    key = normalize_app_name(application)
    if not key or key not in APPLICATIONS_ALLOWLIST:
        available = ", ".join(info["display_name"] for info in APPLICATIONS_ALLOWLIST.values())
        raise SecurityViolation(
            f"Application '{application}' is not in the authorized allowlist. Available applications: {available}"
        )

    app_info = APPLICATIONS_ALLOWLIST[key]
    display_name = app_info["display_name"]
    cmd = app_info["command"]

    try:
        if sys.platform == "win32":
            # Use os.startfile for registered Windows application protocols and executables
            try:
                os.startfile(cmd)
            except Exception:
                # Fallback to direct subprocess invocation with static executable (NO shell=True)
                subprocess.Popen([cmd])
        else:
            subprocess.Popen([cmd])

        SecurityAuditLogger.log_execution("open_application", True, f"Launched {display_name}")
        return f"{display_name} has been launched successfully, sir."
    except Exception as e:
        SecurityAuditLogger.log_execution("open_application", False, f"Failed to launch {display_name}: {e}")
        return f"Unable to open {display_name}: {e}"


def close_application(application: str) -> str:
    """
    Close a running application from the strict allowlist.
    """
    key = normalize_app_name(application)
    if not key or key not in APPLICATIONS_ALLOWLIST:
        raise SecurityViolation(f"Application '{application}' is not authorized to be closed.")

    app_info = APPLICATIONS_ALLOWLIST[key]
    target_exe = app_info["executable"].lower()
    display_name = app_info["display_name"]

    closed_count = 0
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname = proc.info["name"]
            if pname and pname.lower() == target_exe:
                proc.terminate()
                closed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if closed_count > 0:
        SecurityAuditLogger.log_execution("close_application", True, f"Closed {display_name} ({closed_count} instances)")
        return f"{display_name} has been closed, sir."
    else:
        return f"{display_name} does not appear to be running currently, sir."


def is_application_running(application: str) -> str:
    """
    Check if an allowlisted application is currently running.
    """
    key = normalize_app_name(application)
    if not key or key not in APPLICATIONS_ALLOWLIST:
        raise SecurityViolation(f"Application '{application}' is not in the authorized allowlist.")

    app_info = APPLICATIONS_ALLOWLIST[key]
    target_exe = app_info["executable"].lower()
    display_name = app_info["display_name"]

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname = proc.info["name"]
            if pname and pname.lower() == target_exe:
                return f"Yes, {display_name} is currently running."
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return f"No, {display_name} is not currently running."
