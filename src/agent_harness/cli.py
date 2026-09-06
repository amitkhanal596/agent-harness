"""Typer CLI: run / resume / replay / trace / eval."""

from __future__ import annotations

import asyncio

import typer

from .config import get_settings
from .db import get_pool, run_migrations

app = typer.Typer(help="Durable, observable agent-execution engine.", no_args_is_help=True)


@app.command()
def migrate() -> None:
    """Apply pending SQL migrations to the configured database."""

    async def _main() -> None:
        pool = await get_pool()
        try:
            applied = await run_migrations(pool)
        finally:
            await pool.close()
        typer.echo(f"applied: {applied}" if applied else "already up to date")

    asyncio.run(_main())


@app.command()
def config() -> None:
    """Print the effective configuration (secrets redacted)."""
    s = get_settings()
    data = s.model_dump()
    data["anthropic_api_key"] = "<set>" if s.anthropic_api_key else "<unset>"
    for key, value in data.items():
        typer.echo(f"{key}={value}")


if __name__ == "__main__":  # pragma: no cover
    app()
