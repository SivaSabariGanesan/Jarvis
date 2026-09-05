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

    # STT (Speech-to-Text) - Run on CPU to reserve full GPU VRAM for Ollama LLM
    whisper_model_size: str = Field(default="base.en", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="default", alias="WHISPER_COMPUTE_TYPE")
    whisper_cpu_threads: int = Field(default=1, alias="WHISPER_CPU_THREADS")

    # TTS (Text-to-Speech)
    piper_voice: str = Field(default="en_US-lessac-medium", alias="PIPER_VOICE")

    # Wake Word (Local Detection)
    wake_word: str = Field(default="jarvis", alias="WAKE_WORD")
    wake_word_model_path: str = Field(default="", alias="WAKE_WORD_MODEL_PATH")
    wake_word_threshold: float = Field(default=0.5, alias="WAKE_WORD_THRESHOLD")
    wake_word_cooldown: float = Field(default=1.0, alias="WAKE_WORD_COOLDOWN")
    wake_word_activation_response: str = Field(default="Yes, sir?", alias="WAKE_WORD_ACTIVATION_RESPONSE")

    # Memory / Storage
    sqlite_db_path: str = Field(default="data/jarvis_memory.db", alias="SQLITE_DB_PATH")

    # V3 Computer Control & Security
    jarvis_workspace: str = Field(default="D:\\Jarvis", alias="JARVIS_WORKSPACE")
    allowed_directories: list[str] = Field(default_factory=list, alias="JARVIS_ALLOWED_DIRECTORIES")
    tool_timeout_seconds: float = Field(default=15.0, alias="TOOL_TIMEOUT_SECONDS")
    computer_control_enabled: bool = Field(default=True, alias="COMPUTER_CONTROL_ENABLED")

    # V4 Vision & Input Controls
    vision_model: str = Field(default="moondream", alias="VISION_MODEL")
    vision_min_confidence: float = Field(default=0.80, alias="VISION_MIN_CONFIDENCE")
    screenshot_retention_count: int = Field(default=20, alias="SCREENSHOT_RETENTION_COUNT")
    screenshot_save_enabled: bool = Field(default=True, alias="SCREENSHOT_SAVE_ENABLED")
    computer_action_timeout_seconds: float = Field(default=15.0, alias="COMPUTER_ACTION_TIMEOUT_SECONDS")

    # V5 Secure Browser Agent
    browser_action_timeout_seconds: float = Field(default=15.0, alias="BROWSER_ACTION_TIMEOUT_SECONDS")
    browser_domain_policy: str = Field(default="open", alias="BROWSER_DOMAIN_POLICY")
    browser_allowed_domains: list[str] = Field(
        default_factory=lambda: ["google.com", "github.com", "docs.python.org", "python.org", "wikipedia.org"],
        alias="BROWSER_ALLOWED_DOMAINS",
    )
    max_browser_actions_per_task: int = Field(default=20, alias="MAX_BROWSER_ACTIONS_PER_TASK")
    max_browser_retries: int = Field(default=2, alias="MAX_BROWSER_RETRIES")

    @property
    def database_file(self) -> Path:
        path = Path(self.sqlite_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def workspace_path(self) -> Path:
        path = Path(self.jarvis_workspace).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings singleton
settings = JarvisSettings()
