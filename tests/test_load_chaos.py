"""§2.12: load and chaos — contention, pool exhaustion, fuzzing, and failure injection.

The existing suite is strong on unit and integration behaviour but never exercised the
system under contention or partial failure, which is where durable-execution designs
actually break.
"""

from __future__ import annotations

import asyncio
import random
import string
from uuid import uuid4

import asyncpg
import pytest

from agent_harness.agent import Agent, MaxStepsExceeded
from agent_harness.checkpointer import Checkpointer
from agent_harness.executor import Executor, FatalToolError
from agent_harness.guardrails import CircuitBreaker, GuardedLLMClient, RateLimiter
from agent_harness.models import Completion, ToolCall
from agent_harness.tools import build_registry
from agent_harness.tracer import Tracer
from helpers import MockLLMClient
from tests_conftest_helpers import DB_URL  # noqa: F401  (re-exported for clarity)

CONCURRENT_RUNS = 25


def _agent(cp: Checkpointer, tracer: Tracer | None = None, script: list[Completion] | None = None):
    registry = build_registry()
    return Agent(
        llm=MockLLMClient(
            script
            or [
                Completion(stop_reason="end_turn", text="done", prompt_tokens=10,
                           output_tokens=5)
            ]
        ),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        tracer=tracer,
        max_steps=5,
    )


# --- concurrency / contention ---------------------------------------------------------


async def test_many_concurrent_runs_do_not_corrupt_each_other(
    db_pool: asyncpg.Pool
) -> None:
    """25 runs sharing one pool must each land correct, isolated state."""
    cp = Checkpointer(db_pool)

    async def one(index: int) -> str:
        state = await cp.create_run(f"task {index}")
        answer = await _agent(cp).run(state)
        assert answer == "done"
        return str(state.run_id)

    run_ids = await asyncio.gather(*(one(i) for i in range(CONCURRENT_RUNS)))

    assert len(set(run_ids)) == CONCURRENT_RUNS
    rows = await db_pool.fetch(
        "SELECT id, task, status, step_count FROM agent_runs WHERE id = ANY($1::uuid[])",
        [uuid4() if False else __import__("uuid").UUID(r) for r in run_ids],
    )
    assert len(rows) == CONCURRENT_RUNS
    assert all(row["status"] == "completed" for row in rows)
    # Each run kept its own task text — no cross-contamination.
    assert len({row["task"] for row in rows}) == CONCURRENT_RUNS


async def test_concurrent_checkpoints_to_the_same_run_do_not_deadlock(
    db_pool: asyncpg.Pool
) -> None:
    """Racing upserts on one (run_id, step) must serialise, not error or hang."""
    cp = Checkpointer(db_pool)
    state = await cp.create_run("contended")

    async def save(n: int) -> None:
        copy = state.model_copy(deep=True)
        copy.messages.append(copy.messages[0].model_copy(update={"content": f"m{n}"}))
        await cp.save(copy)

    await asyncio.wait_for(asyncio.gather(*(save(i) for i in range(20))), timeout=20)

    count = await db_pool.fetchval(
        "SELECT count(*) FROM run_checkpoints WHERE run_id = $1 AND step_number = 0",
        state.run_id,
    )
    assert count == 1  # upsert held under contention


async def test_concurrent_tracing_totals_are_not_lost(db_pool: asyncpg.Pool) -> None:
    """Racing token/cost increments must sum correctly (no lost updates)."""
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    state = await cp.create_run("totals")
    comp = Completion(stop_reason="end_turn", prompt_tokens=10, output_tokens=5)

    await asyncio.gather(
        *(tracer.llm(state, comp, latency_ms=1, cost_usd=0.001) for _ in range(30))
    )
    totals = await tracer.totals(state.run_id)
    assert totals["total_tokens"] == 30 * 15
    assert float(totals["total_cost_usd"]) == pytest.approx(0.03, abs=1e-6)


async def test_pool_exhaustion_queues_rather_than_failing() -> None:
    """More concurrent work than connections must wait, not raise.

    Builds its own small pool rather than using the `db_pool` fixture, so it needs its
    own skip when Postgres is unavailable.
    """
    try:
        pool = await asyncpg.create_pool(dsn=DB_URL, min_size=1, max_size=2)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"postgres unavailable: {exc}")
    assert pool is not None
    try:
        async def query(n: int) -> int:
            async with pool.acquire() as conn:
                return int(await conn.fetchval("SELECT $1::int", n))

        results = await asyncio.wait_for(
            asyncio.gather(*(query(i) for i in range(20))), timeout=30
        )
        assert sorted(results) == list(range(20))
    finally:
        await pool.close()


async def test_concurrent_tool_fanout_under_load() -> None:
    """Many simultaneous tool calls across many executors stay correct."""
    registry = build_registry()
    executor = Executor(registry, timeout=10.0, max_retries=0)
    calls = [
        ToolCall(id=f"c{i}", name="list_dir", input={"path": "."}) for i in range(50)
    ]
    results = await executor.run_all(calls)
    assert len(results) == 50
    assert [r.call_id for r in results] == [c.id for c in calls]
    assert all(r.error is None for r in results)


# --- chaos / failure injection --------------------------------------------------------


class FlakyLLM(MockLLMClient):
    """Fails a configurable fraction of calls, deterministically seeded."""

    def __init__(self, script: list[Completion], failure_rate: float, seed: int) -> None:
        super().__init__(script, repeat_last=True)
        self.rng = random.Random(seed)
        self.failure_rate = failure_rate
        self.failures = 0

    async def complete(self, messages: object, tools: object) -> Completion:
        if self.rng.random() < self.failure_rate:
            self.failures += 1
            raise ConnectionError("injected transient failure")
        return await super().complete(messages, tools)  # type: ignore[arg-type]


async def test_run_survives_to_a_checkpoint_despite_injected_failures(
    db_pool: asyncpg.Pool
) -> None:
    """When the model flakes, the run fails cleanly and stays resumable."""
    cp = Checkpointer(db_pool)
    state = await cp.create_run("flaky")
    registry = build_registry()
    agent = Agent(
        llm=FlakyLLM(
            [Completion(stop_reason="end_turn", text="done")], failure_rate=1.0, seed=1
        ),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
    )
    with pytest.raises(ConnectionError):
        await agent.run(state)

    # The pre-failure checkpoint is intact and resumable.
    persisted = await cp.latest(state.run_id)
    assert persisted.run_id == state.run_id
    assert await _agent(cp).resume(state.run_id) == "done"


async def test_breaker_contains_a_sustained_outage() -> None:
    """Under a total outage the breaker must stop hammering the provider."""
    class Down:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages: object, tools: object) -> Completion:
            self.calls += 1
            raise ConnectionError("provider down")

    client = Down()
    guarded = GuardedLLMClient(client, RateLimiter(rate=1000, capacity=1000),
                               CircuitBreaker(threshold=5, reset_after=60))

    from agent_harness.guardrails import CircuitOpen

    attempts = 0
    for _ in range(100):
        try:
            await guarded.complete([], [])
        except (ConnectionError, CircuitOpen):
            attempts += 1
    assert attempts == 100
    # 100 attempts, but the provider was only reached 5 times.
    assert client.calls == 5


async def test_fatal_tool_under_heavy_fanout_cancels_everything(
    db_pool: asyncpg.Pool
) -> None:
    from pydantic import BaseModel

    from agent_harness.tools.base import Registry, Tool

    class Empty(BaseModel):
        pass

    class Slow(Tool):
        name, description, InputModel = "slow", "s", Empty

        async def execute(self, args: BaseModel) -> str:
            await asyncio.sleep(10)
            return "never"

    class Boom(Tool):
        name, description, InputModel = "boom", "b", Empty

        async def execute(self, args: BaseModel) -> str:
            await asyncio.sleep(0.05)
            raise FatalToolError("stop everything")

    registry = Registry()
    registry.register(Slow())
    registry.register(Boom())
    executor = Executor(registry, timeout=30.0, max_retries=0)

    calls = [ToolCall(id=f"s{i}", name="slow", input={}) for i in range(30)]
    calls.insert(15, ToolCall(id="boom", name="boom", input={}))

    results = await asyncio.wait_for(executor.run_all(calls), timeout=5)
    assert len(results) == 31
    assert [r.call_id for r in results] == [c.id for c in calls]
    # Every slot reports an error: the 30 siblings were cancelled, and the raiser's own
    # task was torn down by the TaskGroup rather than returning a result.
    assert all(r.error for r in results)
    assert sum(1 for r in results if "cancelled" in (r.error or "")) >= 30


async def test_max_steps_holds_under_a_runaway_model(db_pool: asyncpg.Pool) -> None:
    """A model that never stops must be bounded even with tools succeeding."""
    cp = Checkpointer(db_pool)
    state = await cp.create_run("runaway")
    registry = build_registry()
    agent = Agent(
        llm=MockLLMClient(
            [
                Completion(
                    stop_reason="tool_use",
                    tool_calls=[ToolCall(id="t", name="list_dir", input={"path": "."})],
                )
            ],
            repeat_last=True,
        ),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        max_steps=6,
    )
    with pytest.raises(MaxStepsExceeded):
        await agent.run(state)
    assert state.step_count == 6
    assert (await cp.latest(state.run_id)).status == "failed"


# --- fuzzing ---------------------------------------------------------------------------


def _random_text(rng: random.Random, length: int) -> str:
    alphabet = string.printable + "🙂✓日本語\x00\x1b"
    return "".join(rng.choice(alphabet) for _ in range(length))


@pytest.mark.parametrize("seed", range(8))
async def test_tools_never_raise_out_of_the_executor_on_fuzzed_input(seed: int) -> None:
    """Whatever the model sends, the executor must return data, never propagate."""
    rng = random.Random(seed)
    registry = build_registry()
    executor = Executor(registry, timeout=5.0, max_retries=0)
    names = ["read_file", "list_dir", "grep", "run_shell", "http_get", "nonexistent"]

    for _ in range(25):
        name = rng.choice(names)
        payload: dict[str, object] = {}
        for key in rng.sample(
            ["path", "pattern", "command", "url", "limit", "query"],
            k=rng.randint(0, 3),
        ):
            payload[key] = rng.choice(
                [_random_text(rng, rng.randint(0, 30)), rng.randint(-10**6, 10**6),
                 None, True, [], {}]
            )
        result = await executor.run(ToolCall(id="f", name=name, input=payload))
        # The contract: always a ToolResult, never an exception.
        assert result.call_id == "f"
        assert result.error is not None or isinstance(result.output, str)


@pytest.mark.parametrize("seed", range(5))
async def test_fuzzed_state_survives_a_checkpoint_round_trip(
    db_pool: asyncpg.Pool, seed: int
) -> None:
    """Arbitrary message content must serialise and come back byte-identical."""
    from agent_harness.models import Message

    rng = random.Random(seed)
    cp = Checkpointer(db_pool)
    state = await cp.create_run("fuzz")
    for _ in range(rng.randint(1, 10)):
        state.messages.append(
            Message(role="assistant", content=_random_text(rng, rng.randint(0, 500)))
        )
    state.step_count = rng.randint(0, 50)
    await cp.save(state)

    restored = await cp.latest(state.run_id)
    # NUL cannot be stored in jsonb, so it is escaped on the way in; everything else
    # must round-trip byte-identically.
    expected = [m.content.replace("\x00", "\\u0000") for m in state.messages]
    assert [m.content for m in restored.messages] == expected
    assert restored.step_count == state.step_count


async def test_rate_limiter_under_concurrent_pressure() -> None:
    """Concurrent acquires must not grant more than the bucket allows."""
    limiter = RateLimiter(rate=10.0, capacity=5)
    start = asyncio.get_running_loop().time()
    await asyncio.gather(*(limiter.acquire() for _ in range(10)))
    elapsed = asyncio.get_running_loop().time() - start
    # 5 free, then 5 more at 10/s => at least ~0.5s.
    assert elapsed >= 0.4
