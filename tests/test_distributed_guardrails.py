"""§2.2: guardrail state that survives restarts and coordinates across workers."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import asyncpg
import pytest

from agent_harness.guardrails import (
    CircuitOpen,
    DistributedCircuitBreaker,
    DistributedGuardedLLMClient,
    DistributedRateLimiter,
)
from agent_harness.models import Completion
from helpers import MockLLMClient


def _name(prefix: str) -> str:
    """Unique name per test so parallel/repeated runs never share a row."""
    return f"{prefix}-{uuid4().hex[:8]}"


class AlwaysFails:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: object, tools: object) -> Completion:
        self.calls += 1
        raise ConnectionError("provider down")


# --- circuit breaker -----------------------------------------------------------------


async def test_breaker_trips_at_the_threshold(db_pool: asyncpg.Pool) -> None:
    breaker = DistributedCircuitBreaker(db_pool, name=_name("trip"), threshold=3)
    for _ in range(2):
        await breaker.record_failure()
    assert await breaker.state() == DistributedCircuitBreaker.CLOSED
    await breaker.record_failure()
    assert await breaker.state() == DistributedCircuitBreaker.OPEN
    with pytest.raises(CircuitOpen):
        await breaker.before_call()


async def test_breaker_state_survives_a_restart(db_pool: asyncpg.Pool) -> None:
    """The headline property: a new process must not re-probe a downed provider.

    Two separate instances stand in for two processes; the second never recorded a
    failure itself, yet it must still see the circuit as open.
    """
    name = _name("restart")
    first = DistributedCircuitBreaker(db_pool, name=name, threshold=2, reset_after=60)
    await first.record_failure()
    await first.record_failure()
    assert await first.state() == DistributedCircuitBreaker.OPEN

    # Simulate a redeploy: brand-new object, no in-memory history.
    after_restart = DistributedCircuitBreaker(db_pool, name=name, threshold=2, reset_after=60)
    assert await after_restart.state() == DistributedCircuitBreaker.OPEN
    with pytest.raises(CircuitOpen):
        await after_restart.before_call()


async def test_success_closes_the_breaker_for_everyone(db_pool: asyncpg.Pool) -> None:
    name = _name("close")
    a = DistributedCircuitBreaker(db_pool, name=name, threshold=2, reset_after=60)
    await a.record_failure()
    await a.record_failure()
    assert await a.state() == DistributedCircuitBreaker.OPEN

    await a.record_success()
    b = DistributedCircuitBreaker(db_pool, name=name, threshold=2, reset_after=60)
    assert await b.state() == DistributedCircuitBreaker.CLOSED
    await b.before_call()  # must not raise


async def test_breaker_half_opens_after_the_cooldown(db_pool: asyncpg.Pool) -> None:
    name = _name("cooldown")
    breaker = DistributedCircuitBreaker(db_pool, name=name, threshold=1, reset_after=0.2)
    await breaker.record_failure()
    assert await breaker.state() == DistributedCircuitBreaker.OPEN
    await asyncio.sleep(0.25)
    assert await breaker.state() == DistributedCircuitBreaker.HALF_OPEN
    await breaker.before_call()  # half-open allows a probe


async def test_concurrent_failures_do_not_skip_the_trip(db_pool: asyncpg.Pool) -> None:
    """Read-modify-write under contention must not let workers miss the threshold."""
    name = _name("race")
    breakers = [
        DistributedCircuitBreaker(db_pool, name=name, threshold=5, reset_after=60)
        for _ in range(10)
    ]
    await asyncio.gather(*(b.record_failure() for b in breakers))

    failures = await db_pool.fetchval(
        "SELECT failures FROM circuit_breakers WHERE name = $1", name
    )
    assert failures == 10  # no lost updates
    assert await breakers[0].state() == DistributedCircuitBreaker.OPEN


async def test_breakers_are_isolated_by_name(db_pool: asyncpg.Pool) -> None:
    a = DistributedCircuitBreaker(db_pool, name=_name("iso-a"), threshold=1)
    b = DistributedCircuitBreaker(db_pool, name=_name("iso-b"), threshold=1)
    await a.record_failure()
    assert await a.state() == DistributedCircuitBreaker.OPEN
    assert await b.state() == DistributedCircuitBreaker.CLOSED


def test_breaker_rejects_bad_threshold(db_pool: asyncpg.Pool) -> None:
    with pytest.raises(ValueError):
        DistributedCircuitBreaker(db_pool, threshold=0)


# --- rate limiter --------------------------------------------------------------------


async def test_bucket_allows_a_burst_then_refuses(db_pool: asyncpg.Pool) -> None:
    limiter = DistributedRateLimiter(db_pool, name=_name("burst"), rate=0.1, capacity=3)
    assert await limiter.try_acquire()
    assert await limiter.try_acquire()
    assert await limiter.try_acquire()
    assert not await limiter.try_acquire()  # capacity exhausted, refill is slow


async def test_bucket_is_shared_across_instances(db_pool: asyncpg.Pool) -> None:
    """Two 'workers' must share one budget, not get one each."""
    name = _name("shared")
    worker_a = DistributedRateLimiter(db_pool, name=name, rate=0.01, capacity=2)
    worker_b = DistributedRateLimiter(db_pool, name=name, rate=0.01, capacity=2)

    assert await worker_a.try_acquire()
    assert await worker_b.try_acquire()
    # The bucket is now empty for BOTH; an independent bucket would still allow these.
    assert not await worker_a.try_acquire()
    assert not await worker_b.try_acquire()


async def test_tokens_refill_over_time(db_pool: asyncpg.Pool) -> None:
    limiter = DistributedRateLimiter(db_pool, name=_name("refill"), rate=20.0, capacity=1)
    assert await limiter.try_acquire()
    assert not await limiter.try_acquire()
    await asyncio.sleep(0.2)  # 20/s means a token back in 50ms
    assert await limiter.try_acquire()


async def test_acquire_waits_for_capacity(db_pool: asyncpg.Pool) -> None:
    limiter = DistributedRateLimiter(db_pool, name=_name("wait"), rate=20.0, capacity=1)
    await limiter.acquire()
    start = asyncio.get_running_loop().time()
    await limiter.acquire()  # must block until refilled
    assert asyncio.get_running_loop().time() - start >= 0.03


async def test_concurrent_claims_never_oversubscribe(db_pool: asyncpg.Pool) -> None:
    """Row locking must prevent more grants than the bucket holds."""
    name = _name("oversub")
    limiters = [
        DistributedRateLimiter(db_pool, name=name, rate=0.001, capacity=5)
        for _ in range(20)
    ]
    granted = await asyncio.gather(*(limit.try_acquire() for limit in limiters))
    assert sum(granted) <= 5, f"granted {sum(granted)} tokens from a bucket of 5"


async def test_limiter_rejects_bad_config(db_pool: asyncpg.Pool) -> None:
    with pytest.raises(ValueError):
        DistributedRateLimiter(db_pool, name="x", rate=0, capacity=1)
    with pytest.raises(ValueError):
        await DistributedRateLimiter(db_pool, name=_name("big"), rate=1, capacity=1).try_acquire(9)


# --- guarded client ------------------------------------------------------------------


async def test_guarded_client_fails_fast_once_the_breaker_is_open(
    db_pool: asyncpg.Pool
) -> None:
    breaker = DistributedCircuitBreaker(db_pool, name=_name("guard"), threshold=3,
                                        reset_after=60)
    client = AlwaysFails()
    guarded = DistributedGuardedLLMClient(client, breaker=breaker)

    for _ in range(3):
        with pytest.raises(ConnectionError):
            await guarded.complete([], [])
    assert client.calls == 3

    for _ in range(4):
        with pytest.raises(CircuitOpen):
            await guarded.complete([], [])
    assert client.calls == 3  # the client was never reached again


async def test_guarded_client_resets_on_success(db_pool: asyncpg.Pool) -> None:
    name = _name("reset")
    breaker = DistributedCircuitBreaker(db_pool, name=name, threshold=3, reset_after=60)
    await breaker.record_failure()
    await breaker.record_failure()

    guarded = DistributedGuardedLLMClient(
        MockLLMClient([Completion(stop_reason="end_turn", text="ok")]), breaker=breaker
    )
    await guarded.complete([], [])
    assert await breaker.state() == DistributedCircuitBreaker.CLOSED
