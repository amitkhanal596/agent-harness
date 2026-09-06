"""M7: rate limiter, circuit breaker, and the per-run budget check."""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import asyncpg
import pytest

from agent_harness.agent import Agent, BudgetExceeded
from agent_harness.checkpointer import Checkpointer
from agent_harness.executor import Executor
from agent_harness.guardrails import (
    BudgetExceededError,
    CircuitBreaker,
    CircuitOpen,
    GuardedLLMClient,
    RateLimiter,
    check_budget,
)
from agent_harness.models import Completion, Message, State, ToolCall, ToolSpec
from agent_harness.tools import build_registry
from helpers import MockLLMClient


class AlwaysFailingClient:
    """Simulates an API that is down. Counts how many times it was actually reached."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        self.calls += 1
        raise ConnectionError("api unavailable")


# --- rate limiter -----------------------------------------------------------------


async def test_rate_limiter_allows_burst_up_to_capacity() -> None:
    limiter = RateLimiter(rate=10.0, capacity=3)
    t0 = time.perf_counter()
    for _ in range(3):
        await limiter.acquire()
    assert time.perf_counter() - t0 < 0.05


async def test_rate_limiter_throttles_beyond_capacity() -> None:
    limiter = RateLimiter(rate=20.0, capacity=2)  # refills one token per 50ms
    for _ in range(2):
        await limiter.acquire()
    t0 = time.perf_counter()
    await limiter.acquire()
    assert time.perf_counter() - t0 >= 0.03


async def test_rate_limiter_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        RateLimiter(rate=0, capacity=1)
    with pytest.raises(ValueError):
        await RateLimiter(rate=1, capacity=1).acquire(tokens=5)


async def test_rate_limiter_is_concurrency_safe() -> None:
    limiter = RateLimiter(rate=100.0, capacity=5)
    await asyncio.gather(*(limiter.acquire() for _ in range(5)))


# --- circuit breaker --------------------------------------------------------------


async def test_breaker_trips_after_n_failures_and_fails_fast() -> None:
    client = AlwaysFailingClient()
    guarded = GuardedLLMClient(client, breaker=CircuitBreaker(threshold=3, reset_after=60))

    for _ in range(3):
        with pytest.raises(ConnectionError):
            await guarded.complete([], [])
    assert client.calls == 3
    assert guarded.breaker.state == CircuitBreaker.OPEN

    # subsequent calls fail fast without hitting the client
    for _ in range(4):
        with pytest.raises(CircuitOpen):
            await guarded.complete([], [])
    assert client.calls == 3


async def test_breaker_resets_on_success() -> None:
    breaker = CircuitBreaker(threshold=3)
    guarded = GuardedLLMClient(
        MockLLMClient([Completion(stop_reason="end_turn", text="ok")]), breaker=breaker
    )
    breaker.record_failure()
    breaker.record_failure()
    await guarded.complete([], [])
    assert breaker.failures == 0
    assert breaker.state == CircuitBreaker.CLOSED


async def test_breaker_half_opens_after_cooldown() -> None:
    breaker = CircuitBreaker(threshold=1, reset_after=0.05)
    breaker.record_failure()
    with pytest.raises(CircuitOpen):
        breaker.before_call()
    await asyncio.sleep(0.06)
    breaker.before_call()  # cooldown elapsed -> half open, call allowed
    assert breaker.state == CircuitBreaker.HALF_OPEN


def test_breaker_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(threshold=0)


# --- budget -----------------------------------------------------------------------


def test_check_budget_helper() -> None:
    check_budget(0.5, 1.0)
    with pytest.raises(BudgetExceededError):
        check_budget(1.5, 1.0)


async def test_tiny_budget_aborts_run_and_persists_failed(
    db_pool: asyncpg.Pool
) -> None:
    registry = build_registry()
    cp = Checkpointer(db_pool)
    llm = MockLLMClient(
        [Completion(stop_reason="end_turn", text="hi", prompt_tokens=1_000_000,
                    output_tokens=1_000_000)]
    )
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        max_steps=5,
        budget_usd=0.000001,
    )
    state = await cp.create_run("expensive task")
    with pytest.raises(BudgetExceeded):
        await agent.run(state)

    assert state.status == "failed"
    persisted = await cp.latest(state.run_id)
    assert persisted.status == "failed"
    row = await db_pool.fetchrow("SELECT status FROM agent_runs WHERE id = $1", state.run_id)
    assert row["status"] == "failed"


async def test_budget_not_tripped_under_cap() -> None:
    registry = build_registry()
    llm = MockLLMClient(
        [Completion(stop_reason="end_turn", text="cheap", prompt_tokens=10, output_tokens=5)]
    )
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        budget_usd=1.0,
    )
    state = State(run_id=uuid4(), task="t")
    assert await agent.run(state) == "cheap"
    assert state.status == "completed"


async def test_guarded_client_applies_rate_limit_before_calling() -> None:
    inner = MockLLMClient(
        [Completion(stop_reason="end_turn", text="a"), Completion(stop_reason="end_turn", text="b")]
    )
    guarded = GuardedLLMClient(inner, RateLimiter(rate=20.0, capacity=1), CircuitBreaker())
    await guarded.complete([], [])
    t0 = time.perf_counter()
    await guarded.complete([], [])
    assert time.perf_counter() - t0 >= 0.03
    assert len(inner.calls) == 2


async def test_resume_does_not_reset_the_budget(db_pool: asyncpg.Pool) -> None:
    """§2.3 regression: a resumed run continues against the spend it already incurred.

    Before the fix, `Agent.total_cost_usd` reset to 0 on every `resume()`, so a run
    could be restarted indefinitely and never trip its budget.
    """
    from agent_harness.tracer import Tracer

    registry = build_registry()
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)

    # Step 1: an expensive call that stays under the budget, then a crash.
    class OneExpensiveCallThenDie:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages: object, tools: object) -> Completion:
            self.calls += 1
            if self.calls == 1:
                # $0.60 of a $1.00 budget: 200k input tokens at $3/1M.
                return Completion(
                    stop_reason="tool_use",
                    prompt_tokens=200_000,
                    output_tokens=0,
                    tool_calls=[ToolCall(id="t1", name="list_dir", input={"path": "."})],
                )
            raise RuntimeError("process died")

    agent = Agent(
        llm=OneExpensiveCallThenDie(),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        tracer=tracer,
        max_steps=10,
        budget_usd=1.00,
    )
    state = await cp.create_run("expensive run")
    with pytest.raises(RuntimeError):
        await agent.run(state)

    spent = await tracer.spent_usd(state.run_id)
    assert spent == pytest.approx(0.60, abs=1e-6)

    # Resume with another $0.60 call: cumulative $1.20 must trip the $1.00 budget.
    resumed = Agent(
        llm=MockLLMClient(
            [Completion(stop_reason="end_turn", text="done", prompt_tokens=200_000,
                        output_tokens=0)]
        ),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        tracer=tracer,
        max_steps=10,
        budget_usd=1.00,
    )
    with pytest.raises(BudgetExceeded):
        await resumed.resume(state.run_id)

    row = await db_pool.fetchrow("SELECT status FROM agent_runs WHERE id = $1", state.run_id)
    assert row["status"] == "failed"
