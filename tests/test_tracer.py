"""M5: trace events per step and totals that match the sum of event rows."""

from __future__ import annotations

from pathlib import Path

import asyncpg
import pytest

from agent_harness.agent import Agent
from agent_harness.checkpointer import Checkpointer
from agent_harness.executor import Executor
from agent_harness.models import Completion, ToolCall
from agent_harness.tools import build_registry
from agent_harness.tracer import Tracer
from helpers import MockLLMClient

pytestmark = pytest.mark.asyncio


def _two_step_script(temp_repo: Path) -> list[Completion]:
    return [
        Completion(
            stop_reason="tool_use",
            prompt_tokens=100,
            output_tokens=20,
            tool_calls=[
                ToolCall(id="t1", name="read_file", input={"path": str(temp_repo / "config.txt")}),
                ToolCall(id="t2", name="list_dir", input={"path": str(temp_repo)}),
            ],
        ),
        Completion(stop_reason="end_turn", text="the port is 8080", prompt_tokens=200,
                   output_tokens=30),
    ]


async def _run(pool: asyncpg.Pool, temp_repo: Path) -> tuple[Checkpointer, Tracer, str]:
    registry = build_registry()
    cp, tracer = Checkpointer(pool), Tracer(pool)
    agent = Agent(
        llm=MockLLMClient(_two_step_script(temp_repo)),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=1),
        checkpointer=cp,
        tracer=tracer,
        max_steps=10,
    )
    state = await cp.create_run("read the config")
    await agent.run(state)
    return cp, tracer, str(state.run_id)


async def test_one_llm_event_per_step_and_one_tool_event_per_tool(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> None:
    from uuid import UUID

    _, tracer, run_id = await _run(db_pool, temp_repo)
    events = await tracer.timeline(UUID(run_id))
    kinds = [e["kind"] for e in events]
    assert kinds.count("llm_call") == 2  # one per model call
    assert kinds.count("tool_call") == 2  # one per tool in the single tool step
    assert [e["tool_name"] for e in events if e["kind"] == "tool_call"] == [
        "read_file",
        "list_dir",
    ]


async def test_run_totals_equal_sum_of_event_rows(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> None:
    from uuid import UUID

    _, tracer, run_id = await _run(db_pool, temp_repo)
    events = await tracer.timeline(UUID(run_id))
    totals = await tracer.totals(UUID(run_id))
    event_tokens = sum((e["prompt_tokens"] or 0) + (e["output_tokens"] or 0) for e in events)
    event_cost = sum(float(e["cost_usd"] or 0) for e in events)
    assert totals["total_tokens"] == event_tokens == 350
    assert float(totals["total_cost_usd"]) == pytest.approx(event_cost, abs=1e-6)
    assert totals["status"] == "completed"


async def test_tool_error_is_recorded_on_the_event(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> None:
    from uuid import UUID

    registry = build_registry()
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    agent = Agent(
        llm=MockLLMClient(
            [
                Completion(
                    stop_reason="tool_use",
                    tool_calls=[ToolCall(id="t1", name="ghost_tool", input={})],
                ),
                Completion(stop_reason="end_turn", text="done"),
            ]
        ),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        tracer=tracer,
    )
    state = await cp.create_run("t")
    await agent.run(state)
    events = await tracer.timeline(UUID(str(state.run_id)))
    tool_event = next(e for e in events if e["kind"] == "tool_call")
    assert tool_event["error"] is not None
    assert "unknown tool" in tool_event["error"]


async def test_totals_raises_for_unknown_run(db_pool: asyncpg.Pool) -> None:
    from uuid import uuid4

    with pytest.raises(LookupError):
        await Tracer(db_pool).totals(uuid4())
