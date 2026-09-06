"""M4: durable execution — checkpoint every step, resume, replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import asyncpg
import pytest

from agent_harness.agent import Agent
from agent_harness.checkpointer import Checkpointer
from agent_harness.executor import Executor
from agent_harness.models import Completion, ToolCall
from agent_harness.tools import build_registry
from helpers import MockLLMClient


def _script(temp_repo: Path) -> list[Completion]:
    """Three tool steps then a final answer — long enough to interrupt in the middle."""
    return [
        Completion(
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(id="t1", name="read_file", input={"path": str(temp_repo / "config.txt")})
            ],
        ),
        Completion(
            stop_reason="tool_use",
            tool_calls=[ToolCall(id="t2", name="list_dir", input={"path": str(temp_repo)})],
        ),
        Completion(
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(id="t3", name="grep", input={"pattern": "port", "path": str(temp_repo)})
            ],
        ),
        Completion(stop_reason="end_turn", text="the port is 8080"),
    ]


def _agent(llm: MockLLMClient, cp: Checkpointer) -> Agent:
    registry = build_registry()
    return Agent(
        llm=llm,
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=1),
        checkpointer=cp,
        max_steps=10,
    )


class _Boom(Exception):
    pass


class InterruptingLLM(MockLLMClient):
    """Raises after `crash_after` successful completions, simulating a process death."""

    def __init__(self, script: list[Completion], crash_after: int) -> None:
        super().__init__(script)
        self.crash_after = crash_after
        self.served = 0

    async def complete(self, messages: Any, tools: Any) -> Completion:
        if self.served >= self.crash_after:
            raise _Boom("process died")
        self.served += 1
        return await super().complete(messages, tools)


async def test_create_run_inserts_row_and_seeds_task(db_pool: asyncpg.Pool) -> None:
    cp = Checkpointer(db_pool)
    state = await cp.create_run("do the thing")
    row = await db_pool.fetchrow("SELECT task, status FROM agent_runs WHERE id = $1", state.run_id)
    assert row["task"] == "do the thing"
    assert row["status"] == "running"
    assert state.messages[0].role == "user"
    assert state.messages[0].content == "do the thing"


async def test_resaving_same_step_upserts(db_pool: asyncpg.Pool) -> None:
    cp = Checkpointer(db_pool)
    state = await cp.create_run("t")
    await cp.save(state)
    state.messages.append(state.messages[0].model_copy())
    await cp.save(state)  # same step_number, must not raise
    count = await db_pool.fetchval(
        "SELECT count(*) FROM run_checkpoints WHERE run_id = $1 AND step_number = 0", state.run_id
    )
    assert count == 1
    assert len((await cp.latest(state.run_id)).messages) == 2


async def test_latest_raises_for_unknown_run(db_pool: asyncpg.Pool) -> None:
    from uuid import uuid4

    with pytest.raises(LookupError):
        await Checkpointer(db_pool).latest(uuid4())


async def test_checkpoint_written_every_iteration(db_pool: asyncpg.Pool, temp_repo: Path) -> None:
    cp = Checkpointer(db_pool)
    state = await cp.create_run("read the config")
    await _agent(MockLLMClient(_script(temp_repo)), cp).run(state)
    steps = await cp.list_steps(state.run_id)
    assert [s.step_count for s in steps] == [0, 1, 2, 3]
    assert steps[-1].status == "completed"
    # message counts grow monotonically across replay
    assert [len(s.messages) for s in steps] == sorted(len(s.messages) for s in steps)


async def test_interrupted_run_resumed_matches_uninterrupted(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> None:
    """Golden test: crash after step 2, resume, and land on the same final state."""
    cp = Checkpointer(db_pool)

    clean_state = await cp.create_run("read the config")
    clean_answer = await _agent(MockLLMClient(_script(temp_repo)), cp).run(clean_state)

    crashed_state = await cp.create_run("read the config")
    crashing = InterruptingLLM(_script(temp_repo), crash_after=2)
    with pytest.raises(_Boom):
        await _agent(crashing, cp).run(crashed_state)

    checkpoint = await cp.latest(crashed_state.run_id)
    assert checkpoint.step_count == 2
    assert checkpoint.status == "running"

    remaining = _script(temp_repo)[2:]
    resumed_answer = await _agent(MockLLMClient(remaining), cp).resume(crashed_state.run_id)
    final = await cp.latest(crashed_state.run_id)

    assert resumed_answer == clean_answer == "the port is 8080"
    assert final.status == clean_state.status == "completed"
    assert final.step_count == clean_state.step_count
    assert [(m.role, m.tool_call_id) for m in final.messages] == [
        (m.role, m.tool_call_id) for m in clean_state.messages
    ]
    assert [m.content for m in final.messages] == [m.content for m in clean_state.messages]


async def test_final_output_persisted_on_run_row(db_pool: asyncpg.Pool, temp_repo: Path) -> None:
    cp = Checkpointer(db_pool)
    state = await cp.create_run("read the config")
    answer = await _agent(MockLLMClient(_script(temp_repo)), cp).run(state)
    row = await db_pool.fetchrow(
        "SELECT status, step_count, final_output FROM agent_runs WHERE id = $1", state.run_id
    )
    assert row["status"] == "completed"
    assert row["step_count"] == 3
    assert row["final_output"] == answer


async def test_write_latencies_recorded(db_pool: asyncpg.Pool) -> None:
    cp = Checkpointer(db_pool)
    state = await cp.create_run("t")
    await cp.save(state)
    assert len(cp.write_latencies_ms) >= 2
    assert all(ms >= 0 for ms in cp.write_latencies_ms)
