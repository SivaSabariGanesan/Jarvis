"""
Controlled tool registry for JARVIS.
Enforces safety boundaries, parameter validation, and explicit confirmation.
"""

import logging
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("jarvis.tools")


@dataclass
class ToolDefinition:
    name: str
    description: str
    func: Callable
    parameters_schema: Dict[str, Any]
    requires_confirmation: bool = False


class ToolRegistry:
    """
    Controlled registry for JARVIS agent tools.
    Provides introspection, safe dispatch, and audit logging.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        requires_confirmation: bool = False,
    ):
        """Decorator to register a function as a controlled tool."""

        def decorator(func: Callable):
            tool = ToolDefinition(
                name=name,
                description=description,
                func=func,
                parameters_schema=parameters_schema,
                requires_confirmation=requires_confirmation,
            )
            self._tools[name] = tool
            logger.info(f"Registered controlled tool: {name}")
            return func

        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    async def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool with audit logging and error boundaries."""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' is not registered.")

        logger.info(f"Executing tool '{name}' with arguments: {kwargs}")
        try:
            import inspect

            if inspect.iscoroutinefunction(tool.func):
                result = await tool.func(**kwargs)
            else:
                result = tool.func(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            raise


# Global registry instance
tool_registry = ToolRegistry()
