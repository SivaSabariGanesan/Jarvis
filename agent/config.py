"""
Configuration settings for JARVIS Voice Agent.
Loads settings from environment variables and .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class JarvisSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Assistant Identity
    jarvis_name: str = Field(default="JARVIS", alias="JARVIS_NAME")
    jarvis_user_title: str = Field(default="sir", alias="JARVIS_USER_TITLE")

    # Ollama / LLM
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="llama3.2:3b", alias="OLLAMA_MODEL")

    # LiveKit
    livekit_url: str = Field(default="", alias="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", alias="LIVEKIT_API_SECRET")

    # STT (Speech-to-Text)
    whisper_model_size: str = Field(default="base.en", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="cuda", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="float16", alias="WHISPER_COMPUTE_TYPE")

    # TTS (Text-to-Speech)
    piper_voice: str = Field(default="en_US-lessac-medium", alias="PIPER_VOICE")

    # Memory / Storage
    sqlite_db_path: str = Field(default="data/jarvis_memory.db", alias="SQLITE_DB_PATH")

    @property
    def database_file(self) -> Path:
        path = Path(self.sqlite_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Global settings singleton
settings = JarvisSettings()
