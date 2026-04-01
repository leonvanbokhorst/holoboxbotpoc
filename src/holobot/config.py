"""Central configuration — loaded from env vars and persona YAML."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class SpontaneityLevel(IntEnum):
    OFF = 0
    IDLE_ONLY = 1
    MODERATE = 2
    FREQUENT = 3


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # API keys
    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    # LLM
    llm_model: str = "gpt-4o"

    # Audio
    sample_rate: int = 16_000
    channels: int = 1
    block_duration_ms: int = 30  # VAD frame size

    # Silence / turn-taking
    silence_threshold_s: float = 1.5  # how long silence = end of user turn
    max_user_turn_s: float = 30.0

    # Spontaneity
    spontaneity_level: SpontaneityLevel = SpontaneityLevel.MODERATE
    min_silence_before_impulse: float = 8.0
    impulse_probability: float = 0.3
    max_impulses_per_minute: int = 4
    impulse_check_interval_s: float = 2.0

    # Persona
    persona_path: Path = Path("personas/default.yaml")

    # WebSocket
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765

    # Logging
    log_dir: Path = Path("logs")

    @property
    def block_size(self) -> int:
        return int(self.sample_rate * self.block_duration_ms / 1000)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
