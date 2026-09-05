"""
Base interfaces and types for LLM providers in JARVIS.
Decouples agent logic from specific LLM providers (Ollama, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
from typing import AsyncIterable, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str


class LLMClient(ABC):
    """Abstract interface for LLM operations."""

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Perform a standard non-streaming chat completion."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterable[str]:
        """Perform a streaming chat completion, yielding response text chunks."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the underlying LLM provider service and model are reachable and ready."""
        pass
