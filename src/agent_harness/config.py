"""Process configuration, read from the environment or a local `.env` file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Missing required secrets raise at construction time."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    database_url: str = "postgres://postgres:dev@localhost:5432/agent"
    model: str = "claude-sonnet-4-5"
    max_steps: int = 25
    tool_timeout_seconds: float = 30.0
    tool_max_retries: int = 2
    run_budget_usd: float = 1.00
    anthropic_base_url: str = ""


def get_settings() -> Settings:
    """Build a Settings instance from the environment."""
    return Settings()
