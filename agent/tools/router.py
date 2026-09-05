"""
Tool Router and Intent Dispatcher for JARVIS V3.
Coordinates natural language intents, Ollama tool calls, confirmation flows,
and emergency stop controls.
"""

import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import httpx

from agent.config import settings
from agent.tools.security import (
    security_validator,
    RiskLevel,
    ControlState,
    SecurityViolation,
    ConfirmationRequired,
    SecurityAuditLogger,
)
from agent.tools.registry import tool_registry

logger = logging.getLogger("jarvis.tools.router")


COMPUTER_INTENT_KEYWORDS = {
    "open", "launch", "start", "run", "close", "quit", "exit", "kill", "stop",
    "create", "make", "delete", "remove", "rename", "move", "copy", "read",
    "check", "status", "metrics", "usage", "telemetry", "info", "specs",
    "gpu", "cpu", "ram", "memory", "battery", "disk", "storage",
    "screenshot", "screen", "capture", "grab", "settings",
    "volume", "mute", "unmute", "louder", "quieter", "play", "pause", "music",
    "browse", "browser", "website", "url", "explorer", "folder", "file", "app", "application",
    "taskmgr", "taskmanager", "paint", "notepad", "word", "excel", "calculator", "calc", "chrome", "vscode"
}


def has_computer_intent(text: str) -> bool:
    """Check if the user's input likely pertains to a computer action."""
    words = re.findall(r"\b\w+\b", text.lower())
    return any(w in COMPUTER_INTENT_KEYWORDS for w in words)


class ToolRouter:
    """
    Intelligent and secure router for computer control actions.
    """

    def __init__(self):
        self.pending_confirmation: Optional[Tuple[str, Dict[str, Any]]] = None

    def handle_emergency_stop_check(self, text: str) -> Optional[str]:
        """Check for emergency stop / resume voice or text commands."""
        cleaned = text.strip().lower()

        # Stop patterns
        if cleaned in ("jarvis stop", "emergency stop", "stop computer control", "halt", "pause"):
            security_validator.pause_control(reason="User voice/text emergency stop")
            self.pending_confirmation = None
            return "Emergency stop activated, sir. All computer controls have been paused."

        # Resume patterns
        if cleaned in ("jarvis resume", "resume computer control", "enable computer control"):
            security_validator.resume_control()
            return "Computer control has been resumed, sir. Ready for commands."

        return None

    def handle_pending_confirmation(self, text: str) -> Optional[Tuple[bool, str]]:
        """
        If a high-risk tool is awaiting confirmation, process the user's response.
        Returns: (was_pending, response_text) or None if not pending.
        """
        if not self.pending_confirmation:
            return None

        tool_name, tool_args = self.pending_confirmation
        self.pending_confirmation = None

        if security_validator.is_confirmation_positive(text):
            # Execute with user_confirmed=True
            try:
                import asyncio
                # Run synchronous or asynchronous
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # If inside existing loop
                    task = asyncio.ensure_future(
                        tool_registry.execute_secure(tool_name, tool_args, user_confirmed=True)
                    )
                    # We can await it in async callers or run in new thread
                result = asyncio.run(
                    tool_registry.execute_secure(tool_name, tool_args, user_confirmed=True)
                )
                return True, result
            except Exception as e:
                return True, f"Error executing confirmed action: {e}"
        else:
            return True, "Understood, sir. The action has been cancelled."

    def match_deterministic_intent(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Fast, deterministic pattern matching for unambiguous computer actions.
        Prevents LLM hallucinations on common basic requests.
        """
        t = text.strip().lower()

        # Applications
        if re.match(r"^(open|launch|start)\s+(word|ms word|microsoft word)$", t):
            return "open_application", {"application": "word"}
        if re.match(r"^(open|launch|start)\s+(chrome|google chrome|browser)$", t):
            return "open_application", {"application": "chrome"}
        if re.match(r"^(open|launch|start)\s+(notepad|text editor)$", t):
            return "open_application", {"application": "notepad"}
        if re.match(r"^(open|launch|start)\s+(calc|calculator)$", t):
            return "open_application", {"application": "calculator"}
        if re.match(r"^(open|launch|start)\s+(code|vscode|vs code|visual studio code)$", t):
            return "open_application", {"application": "vscode"}
        if re.match(r"^(open|launch|start)\s+(edge|ms edge|microsoft edge)$", t):
            return "open_application", {"application": "edge"}
        if re.match(r"^(open|launch|start)\s+(paint|ms paint)$", t):
            return "open_application", {"application": "paint"}
        if re.match(r"^(open|launch|start)\s+(spotify)$", t):
            return "open_application", {"application": "spotify"}
        if re.match(r"^(open|launch|start)\s+(explorer|file explorer|files)$", t):
            return "open_application", {"application": "explorer"}
        if re.match(r"^(open|launch|start)\s+(task manager|taskmanager|taskmgr)$", t):
            return "open_application", {"application": "taskmanager"}

        # Close applications
        if re.match(r"^(close|quit|exit|kill)\s+(word|notepad|calculator|chrome|edge|paint|spotify|vscode)$", t):
            m = re.match(r"^(close|quit|exit|kill)\s+(\w+)", t)
            if m:
                return "close_application", {"application": m.group(2)}

        # Windows UI
        if re.match(r"^(open|show|toggle)\s+(start menu|start)$", t):
            return "open_start_menu", {}
        if re.match(r"^(open|show)\s+(settings|windows settings)$", t):
            return "open_settings", {}
        if re.match(r"^(open|show)\s+(task manager|taskmanager)$", t):
            return "open_task_manager", {}

        # System Metrics & Telemetry
        if re.match(r"^(check|what is|show|get|how is)\s+(my\s+)?(gpu|gpu usage|gpu status|graphic card)$", t):
            return "get_gpu_status", {}
        if re.match(r"^(check|what is|show|get|how is)\s+(my\s+)?(cpu|ram|memory|system info|system metrics|battery|specs)$", t):
            return "get_system_metrics", {}
        if re.match(r"^(what|which|show|get)\s+(apps are running|running apps|running applications|applications are open)$", t):
            return "get_running_applications", {}

        # Screenshot
        if re.match(r"^(take|capture|grab)\s+(a\s+)?(screenshot|screen shot|screen capture|screen)$", t):
            return "take_screenshot", {}

        # Media controls
        if re.match(r"^(play|pause|resume|toggle media|toggle play|toggle music)$", t):
            return "media_play_pause", {}
        if re.match(r"^(volume up|turn up the volume|increase volume|louder)$", t):
            return "volume_up", {"steps": 2}
        if re.match(r"^(volume down|turn down the volume|decrease volume|quieter)$", t):
            return "volume_down", {"steps": 2}
        if re.match(r"^(mute|unmute|mute volume|toggle mute|silence)$", t):
            return "volume_mute", {}

        return None

    async def execute_intent(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_confirmed: bool = False,
    ) -> str:
        """Execute a validated tool call through the registry."""
        try:
            return await tool_registry.execute_secure(
                tool_name, tool_args, user_confirmed=user_confirmed
            )
        except ConfirmationRequired as cr:
            # Store pending confirmation
            self.pending_confirmation = (cr.tool_name, cr.arguments)
            return str(cr)
        except SecurityViolation as sv:
            return f"I cannot execute that request, sir. {sv}"
        except Exception as e:
            return f"An error occurred while executing the tool: {e}"

    async def process_user_request(
        self,
        user_text: str,
        ollama_client_func,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """
        Main routing pipeline:
        1. Check emergency stop commands.
        2. Check pending confirmation responses.
        3. Check deterministic fast-path intents.
        4. Query Ollama with tool schemas.
        5. If Ollama returns tool calls -> Execute securely through tool registry.
        6. If Ollama returns direct conversation -> Return spoken reply.
        """
        # 1. Emergency Stop Check
        stop_reply = self.handle_emergency_stop_check(user_text)
        if stop_reply:
            return stop_reply

        # 2. Pending Confirmation Check
        conf_res = self.handle_pending_confirmation(user_text)
        if conf_res is not None:
            _, reply = conf_res
            return reply

        # 3. Deterministic Fast-Path
        fast_intent = self.match_deterministic_intent(user_text)
        if fast_intent:
            tool_name, tool_args = fast_intent
            return await self.execute_intent(tool_name, tool_args)

        # 4. Ollama LLM Reasoning (Attach tool schemas only if user expressed computer action intent)
        try:
            tool_schemas = (
                tool_registry.get_ollama_tools_schema()
                if has_computer_intent(user_text)
                else None
            )
            response_data = await ollama_client_func(
                user_text=user_text,
                history=conversation_history,
                tools=tool_schemas,
            )

            # Check if LLM requested tool calls
            message = response_data.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls and isinstance(tool_calls, list):
                tool_call = tool_calls[0]
                func_data = tool_call.get("function", {})
                tool_name = func_data.get("name")
                tool_args = func_data.get("arguments", {})

                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        tool_args = {}

                if tool_name:
                    logger.info(f"[ROUTER] LLM dispatched tool '{tool_name}' with args: {tool_args}")
                    return await self.execute_intent(tool_name, tool_args)

            # Check if LLM output formatted JSON in message content
            content = message.get("content", "").strip()
            if content.startswith("{") and content.endswith("}"):
                try:
                    parsed = json.loads(content)
                    if "name" in parsed and parsed["name"] in [t.name for t in tool_registry.list_tools()]:
                        t_name = parsed["name"]
                        t_args = parsed.get("parameters", parsed.get("arguments", {}))
                        return await self.execute_intent(t_name, t_args)
                except Exception:
                    pass

            return content if content else "I have received your request, sir."

        except Exception as e:
            logger.error(f"[ROUTER] Error in LLM tool processing: {e}")
            return f"I apologize, sir. An error occurred with the reasoning engine: {e}"


# Global tool router instance
tool_router = ToolRouter()
