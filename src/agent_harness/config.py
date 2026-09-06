"""Process configuration, read from the environment or a local `.env` file."""

from __future__ import annotations

import re
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: A `#` that starts the line or follows whitespace begins an inline comment. A `#`
#: inside a token (a URL fragment, a password) is left alone.
_INLINE_COMMENT = re.compile(r"(?:^|\s)#.*$")


def _strip_inline_comment(value: str) -> str:
    """Drop a trailing ` # comment` from an unquoted .env value.

    `.env.example` documents each variable with an inline comment; python-dotenv keeps
    those on unquoted values, which would otherwise be read as the setting itself.
    """
    return _INLINE_COMMENT.sub("", value).strip()


class Settings(BaseSettings):
    """Runtime settings. Values come from the environment, then `.env`, then defaults."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    database_url: str = "postgres://postgres:dev@localhost:5432/agent"
    model: str = "claude-sonnet-4-5"
    max_steps: int = 25
    tool_timeout_seconds: float = 30.0
    tool_max_retries: int = 2
    run_budget_usd: float = 1.00
    anthropic_base_url: str = ""
    cache_prompt: bool = False
    max_context_tokens: int = 150_000
    keep_recent_messages: int = 8
    otel_exporter: str = "none"
    otel_endpoint: str = ""
    log_level: str = "INFO"
    log_json: bool = True
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3"

    @field_validator("*", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> Any:
        return _strip_inline_comment(value) if isinstance(value, str) else value


def get_settings() -> Settings:
    """Build a Settings instance from the environment."""
    return Settings()
