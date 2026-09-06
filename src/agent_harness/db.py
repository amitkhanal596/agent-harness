"""Postgres connection pooling and the migration runner."""

from __future__ import annotations

from pathlib import Path

import asyncpg

from .config import get_settings


def _find_migrations_dir() -> Path:
    """Locate the migrations directory for both source checkouts and installed packages.

    In a source checkout the SQL lives at `<repo>/migrations`. Once installed it is
    shipped as package data at `agent_harness/migrations`. Resolving only the former
    silently yields a nonexistent path inside a container, so check both.
    """
    packaged = Path(__file__).resolve().parent / "migrations"
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[2] / "migrations"


MIGRATIONS_DIR = _find_migrations_dir()

_APPLIED_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


async def get_pool(dsn: str | None = None) -> asyncpg.Pool:
    """Create an asyncpg pool for `dsn` (defaults to settings.database_url)."""
    url = dsn or get_settings().database_url
    pool = await asyncpg.create_pool(dsn=url, min_size=1, max_size=10)
    if pool is None:  # pragma: no cover - asyncpg only returns None on bad args
        raise RuntimeError(f"could not create connection pool for {url}")
    return pool


async def run_migrations(pool: asyncpg.Pool, directory: Path | None = None) -> list[str]:
    """Apply every `.sql` file in `directory` in filename order, exactly once.

    Already-applied files are skipped via the `schema_migrations` table, so this is
    safe to re-run. Returns the filenames applied by this call.
    """
    path = directory or MIGRATIONS_DIR
    applied: list[str] = []
    async with pool.acquire() as conn:
        await conn.execute(_APPLIED_TABLE)
        for sql_file in sorted(path.glob("*.sql")):
            seen = await conn.fetchval(
                "SELECT 1 FROM schema_migrations WHERE filename = $1", sql_file.name
            )
            if seen:
                continue
            async with conn.transaction():
                await conn.execute(sql_file.read_text())
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)", sql_file.name
                )
            applied.append(sql_file.name)
    return applied
