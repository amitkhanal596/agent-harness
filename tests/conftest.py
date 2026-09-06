"""Shared fixtures: DB pool (skipped without Postgres), MockLLMClient, temp files."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from agent_harness.db import get_pool, run_migrations
from agent_harness.models import Completion, Message, ToolSpec

DB_URL = os.environ.get("TEST_DATABASE_URL", "postgres://postgres:dev@localhost:5432/agent")


class MockLLMClient:
    """Replays a scripted list of Completions; records what it was asked.

    Raises RuntimeError when the script is exhausted so runaway loops fail loudly.
    """

    def __init__(self, script: list[Completion], repeat_last: bool = False) -> None:
        self.script = list(script)
        self.repeat_last = repeat_last
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolSpec]] = []

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        self.calls.append(list(messages))
        self.tools_seen.append(list(tools))
        if not self.script:
            raise RuntimeError("MockLLMClient script exhausted")
        if self.repeat_last and len(self.script) == 1:
            return self.script[0]
        return self.script.pop(0)


@pytest.fixture
def mock_llm() -> type[MockLLMClient]:
    return MockLLMClient


@pytest.fixture
def temp_repo(tmp_path: Path) -> Iterator[Path]:
    """A small tree of files for the filesystem tools."""
    (tmp_path / "config.txt").write_text("host=localhost\nport=8080\ndebug=false\n")
    (tmp_path / "notes.md").write_text("# Notes\nthe port is 8080\nnothing else here\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("port=9090\n")
    yield tmp_path


async def _db_available() -> bool:
    try:
        pool = await get_pool(DB_URL)
    except Exception:
        return False
    await pool.close()
    return True


@pytest_asyncio.fixture
async def db_pool() -> AsyncIterator[object]:
    """An asyncpg pool against a migrated test database; skips if Postgres is absent.

    Start one with:
        docker run --name agent-pg -e POSTGRES_PASSWORD=dev -p 5432:5432 -d postgres:15
    """
    try:
        pool = await get_pool(DB_URL)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"postgres unavailable at {DB_URL}: {exc}")
    await run_migrations(pool)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
def run_id() -> str:
    return str(uuid4())
