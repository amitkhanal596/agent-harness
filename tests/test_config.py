"""M0: the scaffold imports, settings load, and migrations are idempotent."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.config import Settings
from agent_harness.db import MIGRATIONS_DIR, run_migrations


def test_settings_defaults_match_spec() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.max_steps == 25
    assert s.tool_timeout_seconds == 30.0
    assert s.tool_max_retries == 2
    assert s.run_budget_usd == 1.00


def test_migrations_dir_has_init() -> None:
    assert (MIGRATIONS_DIR / "001_init.sql").exists()


@pytest.mark.asyncio
async def test_migrations_are_idempotent(db_pool: object, tmp_path: Path) -> None:
    first = await run_migrations(db_pool)  # type: ignore[arg-type]
    second = await run_migrations(db_pool)  # type: ignore[arg-type]
    assert second == []
    assert isinstance(first, list)


def test_migrations_dir_resolves_and_contains_sql() -> None:
    """Regression (§3.8): MIGRATIONS_DIR must exist for installed packages too.

    It was computed as `parents[2]/migrations`, which is correct in a source checkout
    but resolves to a nonexistent path inside a wheel/container — migrations then
    silently applied nothing and every query failed on a missing table.
    """
    assert MIGRATIONS_DIR.is_dir(), f"{MIGRATIONS_DIR} does not exist"
    assert sorted(p.name for p in MIGRATIONS_DIR.glob("*.sql")), "no .sql files found"
