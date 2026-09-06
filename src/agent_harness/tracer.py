"""Per-event observability: one `run_events` row per LLM call and per tool call."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg

from .models import Completion, State, ToolResult


class Tracer:
    """Writes trace events and accumulates run-level token/cost totals.

    Tracing failures are not swallowed: a broken tracer is a broken run, so DB errors
    propagate to the caller.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def llm(
        self, state: State, comp: Completion, latency_ms: int, cost_usd: float
    ) -> None:
        """Record one model call and add its tokens/cost to the run totals."""
        payload = {
            "text": comp.text,
            "stop_reason": comp.stop_reason,
            "tool_calls": [c.model_dump() for c in comp.tool_calls],
        }
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO run_events (run_id, step_number, kind, output_json, latency_ms,
                                            prompt_tokens, output_tokens, cost_usd)
                    VALUES ($1, $2, 'llm_call', $3::jsonb, $4, $5, $6, $7)
                    """,
                    state.run_id,
                    state.step_count,
                    json.dumps(payload),
                    latency_ms,
                    comp.prompt_tokens,
                    comp.output_tokens,
                    cost_usd,
                )
                await conn.execute(
                    """
                    UPDATE agent_runs
                       SET total_tokens = total_tokens + $2,
                           total_cost_usd = total_cost_usd + $3,
                           updated_at = now()
                     WHERE id = $1
                    """,
                    state.run_id,
                    comp.prompt_tokens + comp.output_tokens,
                    cost_usd,
                )

    async def tool(
        self, state: State, result: ToolResult, tool_name: str, latency_ms: int
    ) -> None:
        """Record one tool call, including its error text when it failed."""
        await self._pool.execute(
            """
            INSERT INTO run_events (run_id, step_number, kind, tool_name, input_json,
                                    output_json, latency_ms, error)
            VALUES ($1, $2, 'tool_call', $3, $4::jsonb, $5::jsonb, $6, $7)
            """,
            state.run_id,
            state.step_count,
            tool_name,
            json.dumps({"call_id": result.call_id}),
            json.dumps({"output": result.output}),
            latency_ms,
            result.error,
        )

    async def timeline(self, run_id: UUID) -> list[dict[str, Any]]:
        """Every event for `run_id` in insertion order, for `agent trace`."""
        rows = await self._pool.fetch(
            """
            SELECT step_number, kind, tool_name, latency_ms, prompt_tokens, output_tokens,
                   cost_usd, error, output_json, created_at
              FROM run_events
             WHERE run_id = $1
             ORDER BY id ASC
            """,
            run_id,
        )
        return [dict(row) for row in rows]

    async def totals(self, run_id: UUID) -> dict[str, Any]:
        """Run summary: status, steps, and the persisted token/cost totals."""
        row = await self._pool.fetchrow(
            """
            SELECT task, status, step_count, total_tokens, total_cost_usd, final_output, error
              FROM agent_runs WHERE id = $1
            """,
            run_id,
        )
        if row is None:
            raise LookupError(f"no such run: {run_id}")
        return dict(row)
