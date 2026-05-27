"""Application configuration loaded from environment variables.

All settings are validated at startup via pydantic. Missing required
values (e.g. ANTHROPIC_API_KEY) raise a clear error rather than failing
mysteriously deeper in the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are read from environment variables and an optional `.env`
    file in the project root. Field names are case-insensitive.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic / Claude
    anthropic_api_key: SecretStr = Field(
        ...,
        description="Anthropic API key. Required.",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-5",
        description="Claude model identifier for coaching feedback.",
    )

    # Vision pipeline
    pose_model_path: Path = Field(
        default=Path("./models/pose_landmarker_full.task"),
        description="Path to MediaPipe pose landmarker .task file.",
    )
    pose_min_detection_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MediaPipe pose detection threshold.",
    )
    pose_min_tracking_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MediaPipe pose tracking threshold.",
    )

    # Paths
    data_dir: Path = Field(default=Path("./data"))
    outputs_dir: Path = Field(default=Path("./outputs"))
    references_dir: Path = Field(default=Path("./references"))

    # Logging
    log_level: str = Field(default="INFO")


# Module-level singleton
settings = Settings()  # type: ignore[call-arg]
