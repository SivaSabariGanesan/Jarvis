"""
Unit and integration smoke tests for JARVIS AI Voice Agent.
"""

import pytest
import asyncio
from pathlib import Path

from agent.config import settings
from agent.prompt import JARVIS_SYSTEM_PROMPT, INITIAL_GREETING
from agent.ai.base import ChatMessage
from agent.ai.ollama_client import OllamaClient
from agent.memory.database import JarvisDatabase
from agent.tools.registry import ToolRegistry


def test_config_and_prompts():
    """Verify settings and prompt strings format properly."""
    assert "JARVIS" in INITIAL_GREETING or "online" in INITIAL_GREETING


def test_sqlite_memory(tmp_path: Path):
    """Verify SQLite database persists sessions, turns, and memories."""
    db_file = tmp_path / "test_memory.db"
    db = JarvisDatabase(db_path=str(db_file))

    # Test session creation
    session_id = "test-session-001"
    db.create_session(session_id, room_name="test-room")

    # Test message logging
    db.log_message(session_id, "user", "Hello Jarvis", latency_ms=120.5)
    db.log_message(session_id, "assistant", "Hello sir. How can I assist?", latency_ms=350.2)

    history = db.get_recent_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello Jarvis"
    assert history[1]["role"] == "assistant"

    # Test key-value memory
    db.set_memory("user_name", "Tony")
    assert db.get_memory("user_name") == "Tony"


def test_tool_registry():
    """Verify controlled tool registration and execution."""
    registry = ToolRegistry()

    @registry.register(
        name="add_numbers",
        description="Add two numbers together",
        parameters_schema={"a": "int", "b": "int"},
    )
    def add(a: int, b: int) -> int:
        return a + b

    tool = registry.get_tool("add_numbers")
    assert tool is not None
    assert tool.description == "Add two numbers together"

    # Async execution
    loop = asyncio.new_event_loop()
    res = loop.run_until_complete(registry.execute("add_numbers", a=5, b=7))
    assert res == 12
    loop.close()


def test_ollama_livekit_llm_factory():
    """Verify Ollama LiveKit LLM adapter builds without exception."""
    client = OllamaClient(host="http://localhost:11434", model="llama3.2:3b")
    llm = client.get_livekit_llm()
    assert llm is not None
