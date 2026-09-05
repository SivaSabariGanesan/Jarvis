"""
Ollama client implementation of LLMClient and LiveKit LLM adapter.
"""

import logging
import json
from typing import AsyncIterable, List, Dict, Any, Optional
import httpx

from livekit.plugins import openai

from agent.ai.base import LLMClient, ChatMessage
from agent.config import settings

logger = logging.getLogger("jarvis.ai.ollama")


class OllamaClient(LLMClient):
    """
    Ollama integration providing health checks, direct chat endpoints,
    and LiveKit LLM factory integration.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout

    async def is_healthy(self) -> bool:
        """Verify Ollama container is responding and the target model exists."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.host}/api/version")
                if res.status_code != 200:
                    logger.warning(f"Ollama health check returned status {res.status_code}")
                    return False

                # Verify target model is available
                models_res = await client.get(f"{self.host}/api/tags")
                if models_res.status_code == 200:
                    data = models_res.json()
                    available_models = [m.get("name") for m in data.get("models", [])]
                    # Check if model or model with tag (e.g. llama3.2:3b vs llama3.2:latest) is present
                    has_model = any(
                        self.model == m or self.model.split(":")[0] == m.split(":")[0]
                        for m in available_models
                    )
                    if not has_model:
                        logger.warning(
                            f"Model '{self.model}' not found in Ollama. Available: {available_models}"
                        )
                        return False
                    return True
                return True
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.host}: {e}")
            return False

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Non-streaming chat completion with Ollama API."""
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.host}/api/chat", json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("message", {}).get("content", "")
        except httpx.HTTPError as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterable[str]:
        """Streaming chat completion yielding tokens as they arrive."""
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.host}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def get_livekit_llm(self) -> openai.LLM:
        """
        Create a LiveKit-compatible OpenAI LLM plugin configured for the local Ollama instance.
        """
        # Ollama exposes OpenAI-compatible endpoint at /v1
        base_url = f"{self.host}/v1"
        logger.info(f"Configuring LiveKit LLM with Ollama model='{self.model}' at {base_url}")
        return openai.LLM.with_ollama(
            model=self.model,
            base_url=base_url,
        )
