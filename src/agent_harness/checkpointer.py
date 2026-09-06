"""Durable run state: create runs, checkpoint every iteration, reload for resume/replay."""

from __future__ import annotations

import time
from uuid import UUID, uuid4

import asyncpg

from .models import Message, State


class Checkpointer:
    """Persists `State` snapshots to Postgres.

    Writes are upserts keyed on (run_id, step_number), so re-saving the same step after
    a resume overwrites rather than erroring. Raises `LookupError` when a run has no
    checkpoint.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self.write_latencies_ms: list[float] = []

    async def create_run(self, task: str) -> State:
        """Insert an `agent_runs` row and return the fresh State seeded with the task."""
        run_id: UUID = await self._pool.fetchval(
            "INSERT INTO agent_runs (id, task) VALUES ($1, $2) RETURNING id", uuid4(), task
        )
        state = State(run_id=run_id, task=task)
        state.messages.append(_user_message(task))
        await self.save(state)
        return state

    async def save(self, state: State) -> None:
        """Checkpoint `state` and refresh the run summary row. Timed for §9 metrics."""
        t0 = time.perf_counter()
        payload = state.model_dump_json()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO run_checkpoints (run_id, step_number, state_json)
                    VALUES ($1, $2, $3::jsonb)
                    ON CONFLICT (run_id, step_number)
                    DO UPDATE SET state_json = EXCLUDED.state_json, created_at = now()
                    """,
                    state.run_id,
                    state.step_count,
                    payload,
                )
                await conn.execute(
                    """
                    UPDATE agent_runs
                       SET status = $2,
                           step_count = $3,
                           final_output = COALESCE($4, final_output),
                           updated_at = now()
                     WHERE id = $1
                    """,
                    state.run_id,
                    state.status,
                    state.step_count,
                    _final_output(state),
                )
        self.write_latencies_ms.append((time.perf_counter() - t0) * 1000)

    async def latest(self, run_id: UUID) -> State:
        """Return the newest checkpoint for `run_id`. Raises LookupError if there is none."""
        row = await self._pool.fetchval(
            """
            SELECT state_json FROM run_checkpoints
             WHERE run_id = $1
             ORDER BY step_number DESC, id DESC
             LIMIT 1
            """,
            run_id,
        )
        if row is None:
            raise LookupError(f"no checkpoints for run {run_id}")
        return State.model_validate_json(row)

    async def list_steps(self, run_id: UUID) -> list[State]:
        """Every checkpoint for `run_id`, ascending by step number (replay order)."""
        rows = await self._pool.fetch(
            """
            SELECT state_json FROM run_checkpoints
             WHERE run_id = $1
             ORDER BY step_number ASC, id ASC
            """,
            run_id,
        )
        return [State.model_validate_json(r["state_json"]) for r in rows]

    async def mark_failed(self, run_id: UUID, error: str) -> None:
        """Record a terminal error on the run summary row."""
        await self._pool.execute(
            "UPDATE agent_runs SET status = 'failed', error = $2, updated_at = now() WHERE id = $1",
            run_id,
            error,
        )


def _user_message(task: str) -> Message:
    return Message(role="user", content=task)


def _final_output(state: State) -> str | None:
    if state.status != "completed":
        return None
    for message in reversed(state.messages):
        if message.role == "assistant" and not message.tool_calls:
            return message.content
    return None
