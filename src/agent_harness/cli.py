"""Typer CLI: run / resume / replay / trace / eval."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar
from uuid import UUID

import asyncpg
import typer

from .agent import Agent, BudgetExceeded, MaxStepsExceeded
from .checkpointer import Checkpointer
from .config import Settings, get_settings
from .context import ContextManager
from .db import get_pool, run_migrations
from .executor import Executor
from .guardrails import CircuitBreaker, GuardedLLMClient, RateLimiter
from .llm import AnthropicClient, LLMClient
from .tools import build_registry
from .tracer import Tracer

app = typer.Typer(help="Durable, observable agent-execution engine.", no_args_is_help=True)

T = TypeVar("T")


def _run_async(coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    return asyncio.run(coro_factory())


def build_llm(settings: Settings) -> LLMClient:
    """Build the guarded Anthropic client from settings."""
    if not settings.anthropic_api_key:
        typer.echo("ANTHROPIC_API_KEY is not set (put it in .env)", err=True)
        raise typer.Exit(code=2)
    client = AnthropicClient(
        model=settings.model,
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url or None,
        cache_prompt=settings.cache_prompt,
    )
    return GuardedLLMClient(client, RateLimiter(rate=2.0, capacity=5), CircuitBreaker())


def build_agent(
    pool: asyncpg.Pool, settings: Settings, llm: LLMClient | None = None
) -> tuple[Agent, Checkpointer, Tracer]:
    """Assemble the agent and its collaborators against a live pool."""
    registry = build_registry(
        shell_timeout=settings.tool_timeout_seconds, http_timeout=settings.tool_timeout_seconds
    )
    checkpointer = Checkpointer(pool)
    tracer = Tracer(pool)
    agent = Agent(
        llm=llm or build_llm(settings),
        registry=registry,
        executor=Executor(
            registry,
            timeout=settings.tool_timeout_seconds,
            max_retries=settings.tool_max_retries,
        ),
        checkpointer=checkpointer,
        tracer=tracer,
        max_steps=settings.max_steps,
        budget_usd=settings.run_budget_usd,
        context=ContextManager(
            max_tokens=settings.max_context_tokens,
            keep_recent_messages=settings.keep_recent_messages,
        ),
    )
    return agent, checkpointer, tracer


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

    _run_async(_main)


@app.command()
def config() -> None:
    """Print the effective configuration (secrets redacted)."""
    settings = get_settings()
    data = settings.model_dump()
    data["anthropic_api_key"] = "<set>" if settings.anthropic_api_key else "<unset>"
    for key, value in data.items():
        typer.echo(f"{key}={value}")


@app.command()
def run(task: str = typer.Argument(..., help="The task for the agent to perform.")) -> None:
    """Start a new run and print the final answer plus the run id."""

    async def _main() -> None:
        settings = get_settings()
        pool = await get_pool(settings.database_url)
        try:
            await run_migrations(pool)
            agent, checkpointer, _ = build_agent(pool, settings)
            state = await checkpointer.create_run(task)
            typer.echo(f"run_id: {state.run_id}")
            try:
                answer = await agent.run(state)
            except (BudgetExceeded, MaxStepsExceeded) as exc:
                await checkpointer.mark_failed(state.run_id, str(exc))
                typer.echo(f"run failed: {exc}", err=True)
                raise typer.Exit(code=1) from None
            typer.echo(f"\n{answer}")
            typer.echo(f"\nrun_id: {state.run_id}")
        finally:
            await pool.close()

    _run_async(_main)


@app.command()
def resume(run_id: str = typer.Argument(..., help="Run id to continue.")) -> None:
    """Resume a run from its newest checkpoint."""

    async def _main() -> None:
        settings = get_settings()
        pool = await get_pool(settings.database_url)
        try:
            agent, checkpointer, _ = build_agent(pool, settings)
            rid = UUID(run_id)
            try:
                answer = await agent.resume(rid)
            except (BudgetExceeded, MaxStepsExceeded) as exc:
                await checkpointer.mark_failed(rid, str(exc))
                typer.echo(f"run failed: {exc}", err=True)
                raise typer.Exit(code=1) from None
            typer.echo(f"\n{answer}")
            typer.echo(f"\nrun_id: {run_id}")
        finally:
            await pool.close()

    _run_async(_main)


@app.command()
def replay(run_id: str = typer.Argument(..., help="Run id to replay.")) -> None:
    """Print every checkpoint for a run in order: step number and message count."""

    async def _main() -> None:
        settings = get_settings()
        pool = await get_pool(settings.database_url)
        try:
            states = await Checkpointer(pool).list_steps(UUID(run_id))
            if not states:
                typer.echo(f"no checkpoints for run {run_id}", err=True)
                raise typer.Exit(code=1)
            typer.echo(f"replay of run {run_id} ({len(states)} checkpoints)")
            for state in states:
                typer.echo(
                    f"  step {state.step_count:>3}  messages={len(state.messages):<4} "
                    f"status={state.status}"
                )
        finally:
            await pool.close()

    _run_async(_main)


@app.command()
def trace(run_id: str = typer.Argument(..., help="Run id to trace.")) -> None:
    """Print per-event latency, tokens and cost, plus the run totals."""

    async def _main() -> None:
        settings = get_settings()
        pool = await get_pool(settings.database_url)
        try:
            tracer = Tracer(pool)
            rid = UUID(run_id)
            events = await tracer.timeline(rid)
            totals = await tracer.totals(rid)
            typer.echo(f"trace for run {run_id}")
            typer.echo(f"task: {totals['task']}")
            typer.echo(
                f"{'step':>4}  {'kind':<10} {'tool':<12} {'latency':>9} "
                f"{'tokens':>10} {'cost_usd':>10}  error"
            )
            event_tokens = 0
            event_cost = 0.0
            for event in events:
                tokens = (event["prompt_tokens"] or 0) + (event["output_tokens"] or 0)
                cost = float(event["cost_usd"] or 0)
                event_tokens += tokens
                event_cost += cost
                typer.echo(
                    f"{event['step_number']:>4}  {event['kind']:<10} "
                    f"{(event['tool_name'] or '-'):<12} {str(event['latency_ms']) + 'ms':>9} "
                    f"{tokens:>10} {cost:>10.6f}  {event['error'] or ''}"
                )
            typer.echo(
                f"\nevents: {len(events)}  tokens: {event_tokens}  cost: ${event_cost:.6f}"
            )
            typer.echo(
                f"run totals: status={totals['status']} steps={totals['step_count']} "
                f"tokens={totals['total_tokens']} cost=${float(totals['total_cost_usd']):.6f}"
            )
        finally:
            await pool.close()

    _run_async(_main)


@app.command("eval")
def eval_command(
    tasks: str = typer.Option("evals/tasks.jsonl", help="Path to the labeled task set."),
    limit: int = typer.Option(0, help="Only run the first N tasks (0 = all)."),
    mock: bool = typer.Option(
        False, "--mock", help="Use the scripted offline model instead of the API."
    ),
) -> None:
    """Run the eval set headlessly and print pass rate, avg cost, and avg latency."""
    from .evals_runner import ScriptedFixtureLLM, run_eval_set

    async def _main() -> None:
        client = ScriptedFixtureLLM() if mock else None
        try:
            report: dict[str, Any] = await run_eval_set(tasks, limit=limit or None, llm=client)
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from None
        typer.echo(report["text"])
        raise typer.Exit(code=0 if report["pass_rate"] == 1.0 else 1)

    _run_async(_main)


if __name__ == "__main__":  # pragma: no cover
    app()
