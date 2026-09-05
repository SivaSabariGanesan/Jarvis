"""
Controlled tool registry for JARVIS V3.
Enforces safety boundaries, parameter validation, risk classification, and audit logging.
"""

import inspect
import asyncio
import logging
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field

from agent.config import settings
from agent.tools.security import (
    RiskLevel,
    ControlState,
    SecurityViolation,
    ConfirmationRequired,
    SecurityAuditLogger,
    security_validator,
)
from agent.tools.applications import open_application, close_application, is_application_running
from agent.tools.windows_ui import (
    open_start_menu,
    open_settings,
    open_task_manager,
    open_file_explorer,
    open_browser_url,
)
from agent.tools.system_info import get_system_metrics, get_gpu_status, get_running_applications
from agent.tools.filesystem import (
    create_folder,
    create_text_file,
    read_text_file,
    rename_file,
    move_file,
    copy_file,
    delete_file,
)
from agent.tools.media import media_play_pause, volume_up, volume_down, volume_mute
from agent.tools.screenshot import take_screenshot
from agent.tools.mouse import move_mouse, click, double_click, right_click, scroll
from agent.tools.keyboard import type_text, press_key, hotkey
from agent.tools.window_info import get_active_window, get_open_windows
from agent.tools.vision import analyze_screen, click_element

logger = logging.getLogger("jarvis.tools.registry")


@dataclass
class ToolDefinition:
    name: str
    description: str
    func: Callable
    parameters_schema: Dict[str, Any]
    risk_level: RiskLevel = RiskLevel.LOW
    category: str = "general"
    requires_confirmation: bool = False


class ToolRegistry:
    """
    Controlled registry for JARVIS V3 computer tools.
    Provides introspection, OpenAI/Ollama tool schemas, safe dispatch, and audit logging.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_v3_tools()

    def register(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        risk_level: RiskLevel = RiskLevel.LOW,
        category: str = "general",
        requires_confirmation: bool = False,
    ):
        """Register a function as a controlled tool."""
        def decorator(func: Callable):
            tool = ToolDefinition(
                name=name,
                description=description,
                func=func,
                parameters_schema=parameters_schema,
                risk_level=risk_level,
                category=category,
                requires_confirmation=requires_confirmation,
            )
            self._tools[name] = tool
            logger.info(f"Registered controlled tool: {name} (Risk: {risk_level.value})")
            return func

        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def get_ollama_tools_schema(self) -> List[Dict[str, Any]]:
        """Generate OpenAI/Ollama compatible function-calling schemas for all registered tools."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                }
            })
        return schemas

    async def execute_secure(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        user_confirmed: bool = False,
    ) -> str:
        """
        Execute a tool through the security validation layer with timeout and audit logging.
        """
        args = arguments or {}
        tool = self._tools.get(name)

        if not tool:
            SecurityAuditLogger.log_validation(name, False, "Unregistered tool")
            raise SecurityViolation(f"Tool '{name}' is not registered.")

        # Check emergency stop status
        if not security_validator.is_enabled():
            SecurityAuditLogger.log_validation(name, False, "Computer control paused / disabled")
            raise SecurityViolation("Computer control is currently paused. Use 'JARVIS RESUME' to re-enable.")

        SecurityAuditLogger.log_request(name, args, tool.risk_level)

        # Enforce confirmation for HIGH risk tools
        if (tool.risk_level == RiskLevel.HIGH or tool.requires_confirmation) and not user_confirmed:
            SecurityAuditLogger.log_validation(name, False, "Requires explicit confirmation")
            raise ConfirmationRequired(
                f"Action '{name}' is high-risk and requires explicit authorization. Do you want me to proceed, sir?",
                tool_name=name,
                arguments=args,
            )

        SecurityAuditLogger.log_validation(name, True)

        # Execute with timeout
        timeout = getattr(settings, "tool_timeout_seconds", 15.0)
        try:
            if inspect.iscoroutinefunction(tool.func):
                result = await asyncio.wait_for(tool.func(**args), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.func, **args), timeout=timeout
                )
            return result
        except asyncio.TimeoutError:
            SecurityAuditLogger.log_execution(name, False, f"Timed out after {timeout}s")
            return f"The tool '{name}' timed out after {timeout} seconds."
        except ConfirmationRequired:
            raise
        except SecurityViolation:
            raise
        except Exception as e:
            SecurityAuditLogger.log_execution(name, False, str(e))
            return f"Error executing tool '{name}': {e}"

    async def execute(self, name: str, **kwargs) -> Any:
        """Backward-compatible execute method."""
        return await self.execute_secure(name, arguments=kwargs)

    def _register_default_v3_tools(self):
        """Register the standard V3 tool suite."""

        # 1. Applications
        self.register(
            name="open_application",
            description="Open an authorized application (word, chrome, notepad, calculator, vscode, edge, paint, spotify, explorer, taskmanager).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "application": {
                        "type": "string",
                        "description": "Name of application to launch (e.g. 'word', 'chrome', 'notepad', 'calculator', 'vscode').",
                    }
                },
                "required": ["application"],
            },
            risk_level=RiskLevel.LOW,
            category="applications",
        )(open_application)

        self.register(
            name="close_application",
            description="Close an active authorized application.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "application": {
                        "type": "string",
                        "description": "Name of application to close (e.g. 'notepad', 'calculator', 'chrome').",
                    }
                },
                "required": ["application"],
            },
            risk_level=RiskLevel.MEDIUM,
            category="applications",
        )(close_application)

        self.register(
            name="is_application_running",
            description="Check if an authorized application is currently running on the computer.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "application": {
                        "type": "string",
                        "description": "Name of application to check (e.g. 'word', 'chrome').",
                    }
                },
                "required": ["application"],
            },
            risk_level=RiskLevel.LOW,
            category="applications",
        )(is_application_running)

        # 2. Windows UI
        self.register(
            name="open_start_menu",
            description="Open the Windows Start menu.",
            parameters_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            category="windows_ui",
        )(open_start_menu)

        self.register(
            name="open_settings",
            description="Open Windows Settings, optionally to a specific page (sound, display, network, bluetooth, apps).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "page": {
                        "type": "string",
                        "description": "Optional settings category (sound, display, network, bluetooth, update, battery).",
                    }
                },
            },
            risk_level=RiskLevel.LOW,
            category="windows_ui",
        )(open_settings)

        self.register(
            name="open_task_manager",
            description="Open Windows Task Manager.",
            parameters_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            category="windows_ui",
        )(open_task_manager)

        self.register(
            name="open_file_explorer",
            description="Open Windows File Explorer at an authorized folder path.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional workspace path to open in Explorer.",
                    }
                },
            },
            risk_level=RiskLevel.LOW,
            category="windows_ui",
        )(open_file_explorer)

        self.register(
            name="open_browser_url",
            description="Open a web URL in the default browser (must be http or https).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target website URL (e.g. 'https://google.com' or 'github.com').",
                    }
                },
                "required": ["url"],
            },
            risk_level=RiskLevel.LOW,
            category="windows_ui",
        )(open_browser_url)

        # 3. System Telemetry
        self.register(
            name="get_system_metrics",
            description="Check computer CPU usage, RAM utilization, available Disk space, and Battery level.",
            parameters_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            category="system_info",
        )(get_system_metrics)

        self.register(
            name="get_gpu_status",
            description="Check NVIDIA GPU usage, VRAM consumption, and temperature.",
            parameters_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            category="system_info",
        )(get_gpu_status)

        self.register(
            name="get_running_applications",
            description="List user-facing applications currently running on Windows.",
            parameters_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            category="system_info",
        )(get_running_applications)

        # 4. Filesystem
        self.register(
            name="create_folder",
            description="Create a new folder in the authorized workspace.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string",
                        "description": "Relative or workspace path of the folder to create.",
                    }
                },
                "required": ["folder_path"],
            },
            risk_level=RiskLevel.MEDIUM,
            category="filesystem",
        )(create_folder)

        self.register(
            name="create_text_file",
            description="Create or write a text file in the authorized workspace.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative or workspace path of the text file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to save into the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
            risk_level=RiskLevel.MEDIUM,
            category="filesystem",
        )(create_text_file)

        self.register(
            name="read_text_file",
            description="Read content from a text file in the authorized workspace.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative or workspace path of the file to read.",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default 100).",
                    },
                },
                "required": ["file_path"],
            },
            risk_level=RiskLevel.LOW,
            category="filesystem",
        )(read_text_file)

        self.register(
            name="rename_file",
            description="Rename a file or folder in the authorized workspace.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Path to the existing file or folder.",
                    },
                    "target_name": {
                        "type": "string",
                        "description": "New name for the file or folder (filename only).",
                    },
                },
                "required": ["source_path", "target_name"],
            },
            risk_level=RiskLevel.MEDIUM,
            category="filesystem",
        )(rename_file)

        self.register(
            name="move_file",
            description="Move a file into a folder within the authorized workspace.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Path to the source file.",
                    },
                    "target_folder": {
                        "type": "string",
                        "description": "Destination directory path.",
                    },
                },
                "required": ["source_path", "target_folder"],
            },
            risk_level=RiskLevel.MEDIUM,
            category="filesystem",
        )(move_file)

        self.register(
            name="copy_file",
            description="Copy a file into a folder within the authorized workspace.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Path to the source file.",
                    },
                    "target_folder": {
                        "type": "string",
                        "description": "Destination directory path.",
                    },
                },
                "required": ["source_path", "target_folder"],
            },
            risk_level=RiskLevel.MEDIUM,
            category="filesystem",
        )(copy_file)

        self.register(
            name="delete_file",
            description="Permanently delete a file in the workspace (HIGH RISK: requires confirmation).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path of the file to permanently remove.",
                    }
                },
                "required": ["file_path"],
            },
            risk_level=RiskLevel.HIGH,
            category="filesystem",
            requires_confirmation=True,
        )(delete_file)

        # 5. Media Controls
        self.register(
            name="media_play_pause",
            description="Play or pause active media playback.",
            parameters_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            category="media",
        )(media_play_pause)

        self.register(
            name="volume_up",
            description="Turn up system audio volume.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of volume step increases (1 to 10).",
                    }
                },
            },
            risk_level=RiskLevel.LOW,
            category="media",
        )(volume_up)

        self.register(
            name="volume_down",
            description="Turn down system audio volume.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of volume step decreases (1 to 10).",
                    }
                },
            },
            risk_level=RiskLevel.LOW,
            category="media",
        )(volume_down)

        self.register(
            name="volume_mute",
            description="Mute or unmute system audio.",
            parameters_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            category="media",
        )(volume_mute)

        # 6. Screenshots
        self.register(
            name="take_screenshot",
            description="Take a screenshot of the computer screen and save it in the workspace.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename for the screenshot.",
                    }
                },
            },
            risk_level=RiskLevel.LOW,
            category="screenshot",
        )(take_screenshot)

        # 7. Mouse Controls
        self.register(
            name="move_mouse",
            description="Move the mouse cursor to specific (x, y) screen coordinates.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Target X pixel coordinate."},
                    "y": {"type": "integer", "description": "Target Y pixel coordinate."},
                },
                "required": ["x", "y"],
            },
            risk_level=RiskLevel.LOW,
            category="mouse",
        )(move_mouse)

        self.register(
            name="click",
            description="Click the primary mouse button (optionally at specified x, y coordinates).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Optional target X pixel coordinate."},
                    "y": {"type": "integer", "description": "Optional target Y pixel coordinate."},
                },
            },
            risk_level=RiskLevel.LOW,
            category="mouse",
        )(click)

        self.register(
            name="double_click",
            description="Double click the primary mouse button (optionally at specified x, y coordinates).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Optional target X pixel coordinate."},
                    "y": {"type": "integer", "description": "Optional target Y pixel coordinate."},
                },
            },
            risk_level=RiskLevel.LOW,
            category="mouse",
        )(double_click)

        self.register(
            name="right_click",
            description="Right click the mouse button (optionally at specified x, y coordinates).",
            parameters_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Optional target X pixel coordinate."},
                    "y": {"type": "integer", "description": "Optional target Y pixel coordinate."},
                },
            },
            risk_level=RiskLevel.LOW,
            category="mouse",
        )(right_click)

        self.register(
            name="scroll",
            description="Scroll the mouse wheel vertically. Positive for up, negative for down.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "integer",
                        "description": "Number of scroll notches (e.g. 5 for up, -5 for down).",
                    }
                },
                "required": ["amount"],
            },
            risk_level=RiskLevel.LOW,
            category="mouse",
        )(scroll)

        # 8. Keyboard Controls
        self.register(
            name="type_text",
            description="Type text string into the currently focused window.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text content to type into focused input.",
                    }
                },
                "required": ["text"],
            },
            risk_level=RiskLevel.LOW,
            category="keyboard",
        )(type_text)

        self.register(
            name="press_key",
            description="Press an authorized keyboard key (e.g. 'enter', 'tab', 'esc', 'backspace', 'up', 'down').",
            parameters_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key name from allowlist (e.g. 'enter', 'tab', 'space', 'esc').",
                    }
                },
                "required": ["key"],
            },
            risk_level=RiskLevel.LOW,
            category="keyboard",
        )(press_key)

        self.register(
            name="hotkey",
            description="Trigger a keyboard shortcut combination (e.g. 'ctrl+c', 'ctrl+v', 'alt+tab', 'win+d').",
            parameters_schema={
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "string",
                        "description": "Key combination (e.g. 'ctrl+c', 'ctrl+v', 'alt+tab').",
                    }
                },
                "required": ["keys"],
            },
            risk_level=RiskLevel.LOW,
            category="keyboard",
        )(hotkey)

        # 9. Window Inspection
        self.register(
            name="get_active_window",
            description="Get the title and process of the currently active desktop window.",
            parameters_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            category="window_info",
        )(get_active_window)

        self.register(
            name="get_open_windows",
            description="List all open user-facing application windows on the desktop.",
            parameters_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            category="window_info",
        )(get_open_windows)

        # 10. Computer Vision & Screen Understanding
        self.register(
            name="analyze_screen",
            description="Capture screen and use local vision to identify visible UI elements, controls, and active windows.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional specific question or element to look for on screen.",
                    }
                },
            },
            risk_level=RiskLevel.LOW,
            category="vision",
        )(analyze_screen)

        self.register(
            name="click_element",
            description="Visually locate a UI element (button, link, search box) by name and click it securely.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "element_label": {
                        "type": "string",
                        "description": "Name or text of the control to click (e.g. 'Search', 'Submit', 'Address bar').",
                    }
                },
                "required": ["element_label"],
            },
            risk_level=RiskLevel.LOW,
            category="vision",
        )(click_element)


# Global tool registry singleton
tool_registry = ToolRegistry()
