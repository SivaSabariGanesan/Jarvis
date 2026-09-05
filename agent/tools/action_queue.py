"""
Controlled multi-step action execution queue for JARVIS V4.
Sequences multi-step tasks with per-step security validation, UI state waiting, and bounded timeouts.
"""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional

from agent.config import settings
from agent.tools.security import security_validator, SecurityViolation, SecurityAuditLogger
from agent.tools.window_info import get_open_windows, get_active_window

logger = logging.getLogger("jarvis.tools.action_queue")


async def wait_for_window(title_keyword: str, timeout_seconds: float = 5.0) -> bool:
    """
    Wait until a window containing title_keyword appears and is visible.
    Bounded by timeout_seconds to avoid indefinite blocking.
    """
    start = time.time()
    kw = title_keyword.strip().lower()

    while time.time() - start < timeout_seconds:
        if not security_validator.is_enabled():
            return False

        # Check active window
        active = get_active_window()
        if kw in active.get("title", "").lower() or kw in active.get("process", "").lower():
            return True

        # Check all open windows
        for win in get_open_windows():
            if kw in win.get("title", "").lower() or kw in win.get("process", "").lower():
                return True

        await asyncio.sleep(0.3)

    return False


class ActionQueue:
    """
    Controlled sequential pipeline for complex computer tasks.
    Each action is independently validated through the security layer before execution.
    """

    def __init__(self, tool_registry_instance):
        self.registry = tool_registry_instance

    async def execute_sequence(
        self, steps: List[Dict[str, Any]], user_confirmed: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a sequence of tool actions.
        Format of each step: {"tool": "tool_name", "args": {...}, "wait_after": 0.5, "wait_for_window": "chrome"}
        """
        results = []

        for idx, step in enumerate(steps):
            if not security_validator.is_enabled():
                SecurityAuditLogger.log_validation(
                    "action_queue", False, "Emergency stop encountered during queue execution."
                )
                return {
                    "success": False,
                    "completed_steps": idx,
                    "total_steps": len(steps),
                    "error": "Computer control paused by emergency stop.",
                    "results": results,
                }

            tool_name = step.get("tool")
            args = step.get("args", {})
            wait_win = step.get("wait_for_window")
            wait_after = float(step.get("wait_after", 0.2))

            if not tool_name:
                continue

            # If step requires waiting for window state first
            if wait_win:
                found = await wait_for_window(wait_win, timeout_seconds=5.0)
                if not found:
                    err = f"Step {idx+1} failed: Window matching '{wait_win}' did not appear in time."
                    logger.warning(err)
                    return {
                        "success": False,
                        "completed_steps": idx,
                        "total_steps": len(steps),
                        "error": err,
                        "results": results,
                    }

            # Execute step through tool registry security pipeline
            try:
                res = await self.registry.execute_secure(
                    tool_name, arguments=args, user_confirmed=user_confirmed
                )
                results.append({"step": idx + 1, "tool": tool_name, "result": res})
            except Exception as e:
                return {
                    "success": False,
                    "completed_steps": idx,
                    "total_steps": len(steps),
                    "error": f"Step {idx+1} ({tool_name}) failed: {e}",
                    "results": results,
                }

            if wait_after > 0:
                await asyncio.sleep(wait_after)

        return {
            "success": True,
            "completed_steps": len(steps),
            "total_steps": len(steps),
            "results": results,
        }
