"""
Main entry point for JARVIS LiveKit Voice Agent worker.
Runs with `uv run python -m agent.main dev` or `uv run python -m agent.main start`.
"""

import sys
import logging
from livekit.agents import cli, WorkerOptions

from agent.config import settings
from agent.agent import entrypoint

# Configure clean, structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("jarvis.main")


def main():
    logger.info(f"==================================================")
    logger.info(f" Starting {settings.jarvis_name} AI Voice Agent (V2)")
    logger.info(f" [INFO] Wake word: {settings.wake_word}")
    logger.info(f" [INFO] STT device: {settings.whisper_device} ({settings.whisper_model_size})")
    logger.info(f" [INFO] STT compute type: {settings.whisper_compute_type}")
    logger.info(f" [INFO] VAD device: cpu")
    logger.info(f" [INFO] TTS device: cpu ({settings.piper_voice})")
    logger.info(f" [INFO] LLM backend: Ollama ({settings.ollama_host})")
    logger.info(f" [INFO] LLM model: {settings.ollama_model} (RTX 4050 GPU)")
    logger.info(f" [INFO] Database: {settings.sqlite_db_path}")
    logger.info(f"==================================================")

    # Initialize LiveKit Worker
    options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=settings.jarvis_name.lower(),
        ws_url=settings.livekit_url if settings.livekit_url else None,
        api_key=settings.livekit_api_key if settings.livekit_api_key else None,
        api_secret=settings.livekit_api_secret if settings.livekit_api_secret else None,
    )

    cli.run_app(options)


if __name__ == "__main__":
    main()
