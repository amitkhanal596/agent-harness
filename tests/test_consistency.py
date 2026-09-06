"""§2.7: detecting drift between trace events and the checkpoints they describe."""

from __future__ import annotations

import asyncpg

from agent_harness.checkpointer import Checkpointer
from agent_harness.models import Completion, ToolResult
from agent_harness.tracer import Tracer


async def test_healthy_run_has_no_orphaned_events(db_pool: asyncpg.Pool) -> None:
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    state = await cp.create_run("t")
    await tracer.llm(
        state, Completion(stop_reason="end_turn", prompt_tokens=1, output_tokens=1),
        latency_ms=1, cost_usd=0.0,
    )
    await cp.save(state)
    assert await tracer.orphaned_events(state.run_id) == []


async def test_event_without_a_checkpoint_is_detected(db_pool: asyncpg.Pool) -> None:
    """Simulates a crash between the trace write and the checkpoint write."""
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    state = await cp.create_run("t")

    # Advance the step but never checkpoint it, then trace against it.
    state.step_count = 7
    await tracer.tool(
        state, ToolResult(call_id="c1", output="x"), tool_name="read_file", latency_ms=1
    )

    orphans = await tracer.orphaned_events(state.run_id)
    assert len(orphans) == 1
    assert orphans[0]["step_number"] == 7
    assert orphans[0]["event_count"] == 1


async def test_orphan_disappears_once_the_checkpoint_lands(db_pool: asyncpg.Pool) -> None:
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    state = await cp.create_run("t")
    state.step_count = 3
    await tracer.tool(
        state, ToolResult(call_id="c1", output="x"), tool_name="grep", latency_ms=1
    )
    assert await tracer.orphaned_events(state.run_id) != []

    await cp.save(state)
    assert await tracer.orphaned_events(state.run_id) == []


async def test_stale_runs_reports_non_terminal_runs(db_pool: asyncpg.Pool) -> None:
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    state = await cp.create_run("abandoned")

    # Backdate the run so it looks idle.
    await db_pool.execute(
        "UPDATE agent_runs SET updated_at = now() - interval '2 hours' WHERE id = $1",
        state.run_id,
    )
    stale = await tracer.stale_runs(idle_seconds=3600)
    ids = {row["id"] for row in stale}
    assert state.run_id in ids


async def test_completed_runs_are_not_stale(db_pool: asyncpg.Pool) -> None:
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    state = await cp.create_run("finished")
    state.status = "completed"
    await cp.save(state)
    await db_pool.execute(
        "UPDATE agent_runs SET updated_at = now() - interval '2 hours' WHERE id = $1",
        state.run_id,
    )
    stale = await tracer.stale_runs(idle_seconds=3600)
    assert state.run_id not in {row["id"] for row in stale}


async def test_recent_running_run_is_not_stale(db_pool: asyncpg.Pool) -> None:
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    state = await cp.create_run("in flight")
    stale = await tracer.stale_runs(idle_seconds=3600)
    assert state.run_id not in {row["id"] for row in stale}


async def test_orphaned_events_without_a_run_filter_scans_everything(
    db_pool: asyncpg.Pool
) -> None:
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    state = await cp.create_run("t")
    state.step_count = 11
    await tracer.tool(
        state, ToolResult(call_id="c1", output="x"), tool_name="list_dir", latency_ms=1
    )
    all_orphans = await tracer.orphaned_events()
    assert any(
        row["run_id"] == state.run_id and row["step_number"] == 11 for row in all_orphans
    )
