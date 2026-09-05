"""
Goal-Oriented Browser Planner & Execution Loop for JARVIS V5.
Decomposes high-level browser tasks into structured tool steps, enforcing the
Observe -> Act -> Verify cycle, credential prompt protection, rate limiting, and Takeover mode.
"""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional

from agent.config import settings
from agent.tools.security import (
    security_validator,
    RiskLevel,
    SecurityViolation,
    ConfirmationRequired,
    SecurityAuditLogger,
)
from agent.tools.registry import tool_registry
from agent.tools.browser import get_browser_state
from agent.tools.screenshot import capture_screen_image

logger = logging.getLogger("jarvis.ai.browser_planner")

REGISTERED_BROWSER_TOOL_NAMES = {
    "open_browser",
    "open_url",
    "browser_search",
    "browser_go_back",
    "browser_go_forward",
    "browser_refresh",
    "get_browser_state",
    "click_element",
    "type_text",
    "press_key",
    "hotkey",
    "analyze_screen",
    "take_screenshot",
}


class BrowserPlanner:
    """
    Structured planner and Observe -> Act -> Verify executor for web tasks.
    """

    def __init__(self):
        self._takeover_active: bool = False
        self._action_count: int = 0
        self._max_actions = getattr(settings, "max_browser_actions_per_task", 20)

    @property
    def is_takeover_active(self) -> bool:
        return self._takeover_active

    def pause_for_takeover(self, reason: str = "User requested takeover") -> str:
        """Pause browser automation and yield control to the user."""
        self._takeover_active = True
        msg = "Browser automation paused. You have control, sir."
        SecurityAuditLogger.log_validation("takeover_mode", True, f"Paused: {reason}")
        return msg

    async def resume_after_takeover(self) -> str:
        """Inspect fresh screen state and resume automation."""
        self._takeover_active = False
        state = get_browser_state()
        summary = f"Browser control resumed. Current active window: '{state.get('window_title', 'Desktop')}'. Ready for next instruction."
        SecurityAuditLogger.log_validation("takeover_mode", True, "Resumed after fresh observation")
        return summary

    def create_deterministic_plan(self, user_text: str) -> Optional[Dict[str, Any]]:
        """
        Fast, deterministic planner for common browser requests.
        Ensures strict tool names and parameters without LLM hallucinations.
        """
        t = user_text.strip().lower()

        # Open browser & search
        import re
        m_search = re.match(r"^(?:open\s+(?:chrome|google chrome|browser)\s+and\s+)?search\s+(?:for\s+)?(.+)$", t)
        if m_search:
            query = m_search.group(1).strip()
            return {
                "goal": f"Search for '{query}'",
                "steps": [
                    {"tool": "open_browser", "args": {"browser_name": "chrome"}, "risk": "LOW"},
                    {"tool": "browser_search", "args": {"query": query}, "risk": "LOW"},
                ],
            }

        # Open specific URL
        m_url = re.match(r"^(?:open|go to|navigate to)\s+(https?://\S+|www\.\S+|\S+\.\S+)(?:.*)?$", t)
        if m_url:
            url = m_url.group(1).strip()
            return {
                "goal": f"Navigate to '{url}'",
                "steps": [
                    {"tool": "open_url", "args": {"url": url}, "risk": "LOW"},
                ],
            }

        # Navigation controls
        if t in ("go back", "browser go back", "back"):
            return {
                "goal": "Go back in browser",
                "steps": [{"tool": "browser_go_back", "args": {}, "risk": "LOW"}],
            }
        if t in ("go forward", "browser go forward", "forward"):
            return {
                "goal": "Go forward in browser",
                "steps": [{"tool": "browser_go_forward", "args": {}, "risk": "LOW"}],
            }
        if t in ("refresh", "refresh page", "refresh the page", "reload"):
            return {
                "goal": "Refresh browser page",
                "steps": [{"tool": "browser_refresh", "args": {}, "risk": "LOW"}],
            }

        return None

    def validate_plan(self, plan: Dict[str, Any]) -> None:
        """
        Strictly validate planner output:
        1. Only registered, authorized tool names.
        2. Parameter structure and types.
        3. Rate limits on total step count.
        """
        if not isinstance(plan, dict):
            raise SecurityViolation("Invalid plan structure: must be a dictionary.")

        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            raise SecurityViolation("Invalid plan structure: 'steps' must be a list.")

        if len(steps) > self._max_actions:
            raise SecurityViolation(
                f"Plan step count ({len(steps)}) exceeds maximum allowed ({self._max_actions})."
            )

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                raise SecurityViolation(f"Step {idx+1} is malformed.")

            tool_name = step.get("tool")
            if not tool_name or tool_name not in REGISTERED_BROWSER_TOOL_NAMES:
                raise SecurityViolation(
                    f"Step {idx+1} requested unauthorized or unknown tool '{tool_name}'."
                )

            # Check tool exists in registry
            if not tool_registry.get_tool(tool_name):
                raise SecurityViolation(f"Step {idx+1} tool '{tool_name}' is not registered.")

    async def execute_plan(
        self, plan: Dict[str, Any], user_confirmed: bool = False
    ) -> str:
        """
        Execute the plan adhering to the Observe -> Act -> Verify cycle.
        """
        if not security_validator.is_enabled():
            raise SecurityViolation("Computer control is paused.")

        if self._takeover_active:
            return "Browser automation is currently paused in Takeover mode. Say 'JARVIS CONTINUE' to resume."

        self.validate_plan(plan)
        goal = plan.get("goal", "Browser Task")
        steps = plan.get("steps", [])

        print(f"\033[94m[PLANNER] Goal: {goal} ({len(steps)} steps)\033[0m", flush=True)

        for idx, step in enumerate(steps):
            # 1. Emergency stop check
            if not security_validator.is_enabled():
                return "Browser automation halted by emergency stop."

            if self._takeover_active:
                return "Browser automation paused for user takeover."

            # 2. Rate limiting check
            self._action_count += 1
            if self._action_count > self._max_actions:
                return "The browser task exceeded the safe action limit, so I stopped."

            tool_name = step.get("tool")
            args = step.get("args", {})

            # 3. Check for credential/login prompts before acting
            active_state = get_browser_state()
            active_title = str(active_state.get("window_title", ""))
            if security_validator.is_login_or_credential_prompt(active_title):
                self.pause_for_takeover(reason="Credential prompt detected")
                return "The website is asking for credentials. Please take over."

            # 4. ACT: Execute step through security registry
            print(f"\033[90m[PLANNER] Step {idx+1}/{len(steps)}: {tool_name}\033[0m", flush=True)
            try:
                res = await tool_registry.execute_secure(
                    tool_name, arguments=args, user_confirmed=user_confirmed
                )
            except ConfirmationRequired:
                raise
            except SecurityViolation as sv:
                return f"Browser step {idx+1} blocked by security: {sv}"
            except Exception as e:
                return f"Browser step {idx+1} ({tool_name}) failed: {e}"

            # 5. OBSERVE & VERIFY
            await asyncio.sleep(0.4)
            post_state = get_browser_state()
            if not post_state.get("running") and tool_name not in ("open_browser", "open_url"):
                logger.warning(f"[BROWSER] Post-step verification failed for step {idx+1}")

        self._action_count = 0
        return f"Completed: {goal}."


# Global browser planner instance
browser_planner = BrowserPlanner()
