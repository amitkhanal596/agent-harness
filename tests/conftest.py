"""Shared fixtures: DB pool (skipped without Postgres), MockLLMClient, temp files."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from agent_harness.db import get_pool, run_migrations
from helpers import MockLLMClient

DB_URL = os.environ.get("TEST_DATABASE_URL", "postgres://postgres:dev@localhost:5432/agent")


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


VECTOR_DB_URL = os.environ.get("TEST_VECTOR_DATABASE_URL", DB_URL)


@pytest_asyncio.fixture
async def vector_pool() -> AsyncIterator[object]:
    """A pool against a pgvector-enabled database; skips when the extension is absent.

    Retrieval is optional, so its tests must not fail a run on a plain Postgres. Start a
    capable server with:
        docker run -d -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=agent \
            -p 5433:5432 pgvector/pgvector:pg17
        TEST_VECTOR_DATABASE_URL=postgres://postgres:dev@localhost:5433/agent
    """
    try:
        pool = await get_pool(VECTOR_DB_URL)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"postgres unavailable at {VECTOR_DB_URL}: {exc}")
    try:
        await run_migrations(pool)
        has_vector = await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        if not has_vector:
            pytest.skip("pgvector extension not installed on this server")
        yield pool
    finally:
        await pool.close()
