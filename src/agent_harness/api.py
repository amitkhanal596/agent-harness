"""HTTP service over the same engine the CLI drives.

The CLI opens a connection pool per invocation, which is fine interactively but means
nothing in the repo demonstrates pool reuse under concurrent load — the actual operating
mode of anything running this as a service. This module is that missing piece: one pool
held for the process lifetime, shared by every request.

It deliberately adds no logic of its own. `Agent`, `Checkpointer`, and `Tracer` are used
exactly as the CLI uses them, which is the point — it demonstrates that the engine was
genuinely decoupled from its interface rather than entangled with the CLI.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .agent import Agent, BudgetExceeded, MaxStepsExceeded, RunPaused
from .checkpointer import Checkpointer
from .config import Settings, get_settings
from .context import ContextManager
from .db import get_pool, run_migrations
from .executor import Executor
from .llm import LLMClient
from .logging import configure_logging, get_logger
from .otel import configure_tracing
from .tools import build_registry
from .tracer import Tracer

logger = get_logger(__name__)


class RunRequest(BaseModel):
    task: str = Field(min_length=1, description="What the agent should do.")


class RunResponse(BaseModel):
    run_id: UUID
    status: str
    output: str | None = None
    error: str | None = None
    steps: int = 0


class RunSummary(BaseModel):
    run_id: UUID
    task: str
    status: str
    steps: int
    total_tokens: int
    total_cost_usd: float
    final_output: str | None = None
    error: str | None = None


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    pool: asyncpg.Pool | None = None,
) -> FastAPI:
    """Build the ASGI app.

    `llm` and `pool` are injectable so tests can drive the real routes with a scripted
    model and an existing pool, without network access.
    """
    cfg = settings or get_settings()
    injected_pool = pool

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(level=cfg.log_level, json_format=cfg.log_json)
        configure_tracing(exporter=cfg.otel_exporter, endpoint=cfg.otel_endpoint or None)
        # One pool for the process, not one per request — the whole point of §2.6.
        app.state.pool = injected_pool or await get_pool(cfg.database_url)
        app.state.owns_pool = injected_pool is None
        await run_migrations(app.state.pool)
        logger.info("api started")
        try:
            yield
        finally:
            if app.state.owns_pool:
                await app.state.pool.close()
            logger.info("api stopped")

    app = FastAPI(title="agent-harness", version="0.1.0", lifespan=lifespan)

    def _build(pool_: asyncpg.Pool) -> tuple[Agent, Checkpointer, Tracer]:
        registry = build_registry(
            shell_timeout=cfg.tool_timeout_seconds, http_timeout=cfg.tool_timeout_seconds
        )
        checkpointer = Checkpointer(pool_)
        tracer = Tracer(pool_, model=cfg.model)
        if llm is None:  # pragma: no cover - exercised only with a real API key
            from .cli import build_llm

            client = build_llm(cfg, pool_)
        else:
            client = llm
        agent = Agent(
            llm=client,
            registry=registry,
            executor=Executor(
                registry, timeout=cfg.tool_timeout_seconds, max_retries=cfg.tool_max_retries
            ),
            checkpointer=checkpointer,
            tracer=tracer,
            max_steps=cfg.max_steps,
            budget_usd=cfg.run_budget_usd,
            context=ContextManager(
                max_tokens=cfg.max_context_tokens,
                keep_recent_messages=cfg.keep_recent_messages,
            ),
        )
        return agent, checkpointer, tracer

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """Liveness plus a real database round-trip, so it fails when Postgres does."""
        try:
            await request.app.state.pool.fetchval("SELECT 1")
        except Exception as exc:  # noqa: BLE001 - the endpoint reports, never raises
            raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from None
        return {"status": "ok"}

    @app.post("/runs", response_model=RunResponse, status_code=201)
    async def create_run(request: Request, body: RunRequest) -> RunResponse:
        """Start a run and return its outcome.

        Synchronous by design: the CLI semantics are "run to completion", and a job
        queue would be a different (larger) design decision than this endpoint implies.
        Long tasks should poll `GET /runs/{id}` from a background submission instead.
        """
        agent, checkpointer, _ = _build(request.app.state.pool)
        state = await checkpointer.create_run(body.task)
        try:
            output = await agent.run(state)
        except (BudgetExceeded, MaxStepsExceeded, RunPaused) as exc:
            await checkpointer.mark_failed(state.run_id, str(exc))
            return RunResponse(
                run_id=state.run_id, status=state.status, error=str(exc),
                steps=state.step_count,
            )
        return RunResponse(
            run_id=state.run_id, status="completed", output=output, steps=state.step_count
        )

    @app.get("/runs/{run_id}", response_model=RunSummary)
    async def get_run(request: Request, run_id: UUID) -> RunSummary:
        """Current state of a run."""
        tracer = Tracer(request.app.state.pool)
        try:
            totals = await tracer.totals(run_id)
        except LookupError:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}") from None
        return RunSummary(
            run_id=run_id,
            task=totals["task"],
            status=totals["status"],
            steps=totals["step_count"],
            total_tokens=totals["total_tokens"],
            total_cost_usd=float(totals["total_cost_usd"]),
            final_output=totals["final_output"],
            error=totals["error"],
        )

    @app.get("/runs/{run_id}/trace")
    async def get_trace(request: Request, run_id: UUID) -> dict[str, Any]:
        """Per-event timeline plus run totals."""
        tracer = Tracer(request.app.state.pool)
        try:
            totals = await tracer.totals(run_id)
        except LookupError:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}") from None
        events = await tracer.timeline(run_id)
        return {
            "run_id": str(run_id),
            "status": totals["status"],
            "total_tokens": totals["total_tokens"],
            "total_cost_usd": float(totals["total_cost_usd"]),
            "events": [
                {
                    "step": e["step_number"],
                    "kind": e["kind"],
                    "tool": e["tool_name"],
                    "latency_ms": e["latency_ms"],
                    "prompt_tokens": e["prompt_tokens"],
                    "output_tokens": e["output_tokens"],
                    "cost_usd": float(e["cost_usd"]) if e["cost_usd"] is not None else None,
                    "error": e["error"],
                }
                for e in events
            ],
        }

    @app.get("/runs/{run_id}/checkpoints")
    async def get_checkpoints(request: Request, run_id: UUID) -> dict[str, Any]:
        """Checkpoint list, the HTTP equivalent of `agent replay`."""
        states = await Checkpointer(request.app.state.pool).list_steps(run_id)
        if not states:
            raise HTTPException(status_code=404, detail=f"no checkpoints for run {run_id}")
        return {
            "run_id": str(run_id),
            "checkpoints": [
                {"step": s.step_count, "messages": len(s.messages), "status": s.status}
                for s in states
            ],
        }

    @app.post("/runs/{run_id}/resume", response_model=RunResponse)
    async def resume_run(request: Request, run_id: UUID) -> RunResponse:
        """Continue a paused or interrupted run from its newest checkpoint."""
        agent, checkpointer, _ = _build(request.app.state.pool)
        try:
            output = await agent.resume(run_id)
        except LookupError:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}") from None
        except (BudgetExceeded, MaxStepsExceeded, RunPaused) as exc:
            await checkpointer.mark_failed(run_id, str(exc))
            return RunResponse(run_id=run_id, status="failed", error=str(exc))
        state = await checkpointer.latest(run_id)
        return RunResponse(
            run_id=run_id, status="completed", output=output, steps=state.step_count
        )

    @app.get("/health/consistency")
    async def consistency(request: Request) -> dict[str, Any]:
        """`agent doctor` over HTTP: orphaned events and stalled runs."""
        tracer = Tracer(request.app.state.pool)
        orphans = await tracer.orphaned_events()
        stale = await tracer.stale_runs()
        return {
            "orphaned_events": [
                {"run_id": str(o["run_id"]), "step": o["step_number"],
                 "events": o["event_count"]}
                for o in orphans
            ],
            "stale_runs": [
                {"run_id": str(s["id"]), "status": s["status"], "steps": s["step_count"]}
                for s in stale
            ],
            "healthy": not orphans and not stale,
        }

    return app


def main() -> None:  # pragma: no cover - process entrypoint
    """Run the service with uvicorn."""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        create_app(cfg), host=cfg.api_host, port=cfg.api_port, log_config=None
    )


if __name__ == "__main__":  # pragma: no cover
    main()
