"""Rate limiting, circuit breaking, and the per-run budget check."""

from __future__ import annotations

import asyncio
import time

import asyncpg

from .models import Completion, Message, ToolSpec


class RateLimiter:
    """Token bucket. `acquire()` waits until a token is available.

    `rate` tokens are added per second up to `capacity`; the bucket starts full.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until `tokens` are available, then consume them."""
        if tokens > self.capacity:
            raise ValueError("requested more tokens than the bucket can hold")
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                await asyncio.sleep((tokens - self._tokens) / self.rate)


class CircuitOpen(Exception):
    """The breaker is open: the call was rejected without reaching the client."""


class CircuitBreaker:
    """Trips open after `threshold` consecutive failures; half-opens after `reset_after`.

    While open, `before_call()` raises `CircuitOpen` so callers fail fast.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, threshold: int = 5, reset_after: float = 30.0) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold
        self.reset_after = reset_after
        self.failures = 0
        self.state = self.CLOSED
        self._opened_at = 0.0

    def before_call(self) -> None:
        """Raise `CircuitOpen` when the breaker is open and the cooldown has not elapsed."""
        if self.state == self.OPEN:
            if time.monotonic() - self._opened_at >= self.reset_after:
                self.state = self.HALF_OPEN
            else:
                raise CircuitOpen(
                    f"circuit open after {self.failures} consecutive failures; "
                    f"retry in {self.reset_after - (time.monotonic() - self._opened_at):.1f}s"
                )

    def record_success(self) -> None:
        """Reset the breaker."""
        self.failures = 0
        self.state = self.CLOSED

    def record_failure(self) -> None:
        """Count a failure and trip open at the threshold."""
        self.failures += 1
        if self.failures >= self.threshold:
            self.state = self.OPEN
            self._opened_at = time.monotonic()


class BudgetExceededError(Exception):
    """Raised by `check_budget` when spend passes the configured cap."""


def check_budget(spent_usd: float, budget_usd: float) -> None:
    """Raise `BudgetExceededError` if `spent_usd` is over `budget_usd`."""
    if spent_usd > budget_usd:
        raise BudgetExceededError(
            f"run cost ${spent_usd:.6f} exceeds budget ${budget_usd:.6f}"
        )


class GuardedLLMClient:
    """Wraps an LLMClient with a rate limiter and a circuit breaker.

    Client exceptions are counted by the breaker and re-raised; once open, calls fail
    fast with `CircuitOpen` without touching the wrapped client.
    """

    def __init__(
        self,
        client: object,
        limiter: RateLimiter | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._client = client
        self.limiter = limiter
        self.breaker = breaker or CircuitBreaker()

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        """Rate-limit, check the breaker, then delegate."""
        self.breaker.before_call()
        if self.limiter is not None:
            await self.limiter.acquire()
        try:
            completion: Completion = await self._client.complete(messages, tools)  # type: ignore[attr-defined]
        except Exception:
            self.breaker.record_failure()
            raise
        self.breaker.record_success()
        return completion


class DistributedRateLimiter:
    """Token bucket shared across processes via Postgres.

    The in-process `RateLimiter` gives each worker its own bucket, so N workers hit the
    provider at N times the configured rate. This one keeps a single row per bucket and
    takes a row lock to claim tokens, so the limit holds for the fleet.

    Refill is computed from elapsed time on each read, so there is no background job.
    Every claim costs a round-trip; use the in-process limiter when one worker is the
    whole deployment.
    """

    def __init__(
        self, pool: asyncpg.Pool, name: str, rate: float, capacity: float
    ) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self._pool = pool
        self.name = name
        self.rate = rate
        self.capacity = capacity

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        """Claim `tokens` if available. Returns False rather than waiting."""
        if tokens > self.capacity:
            raise ValueError("requested more tokens than the bucket can hold")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT tokens, updated_at FROM rate_limit_buckets"
                    " WHERE name = $1 FOR UPDATE",
                    self.name,
                )
                if row is None:
                    # First use: start full, minus what this call takes.
                    await conn.execute(
                        "INSERT INTO rate_limit_buckets (name, tokens) VALUES ($1, $2)"
                        " ON CONFLICT (name) DO NOTHING",
                        self.name,
                        self.capacity - tokens,
                    )
                    return True

                elapsed = await conn.fetchval(
                    "SELECT EXTRACT(EPOCH FROM (now() - $1::timestamptz))", row["updated_at"]
                )
                available = min(self.capacity, row["tokens"] + float(elapsed) * self.rate)
                if available < tokens:
                    return False
                await conn.execute(
                    "UPDATE rate_limit_buckets SET tokens = $2, updated_at = now()"
                    " WHERE name = $1",
                    self.name,
                    available - tokens,
                )
                return True

    async def acquire(self, tokens: float = 1.0, poll_interval: float = 0.05) -> None:
        """Wait until `tokens` can be claimed.

        Polls rather than waiting on an event: the tokens are replenished by the passage
        of time and by *other processes* releasing capacity, neither of which an
        in-process `asyncio.Event` can observe. (Postgres LISTEN/NOTIFY could, but that
        is a heavier dependency than this contention profile justifies.)
        """
        while not await self.try_acquire(tokens):  # noqa: ASYNC110 - cross-process wait
            await asyncio.sleep(poll_interval)


class DistributedCircuitBreaker:
    """Circuit breaker whose state lives in Postgres, shared by every worker.

    The in-process breaker forgets it was open whenever the process restarts, so a
    provider outage gets re-probed by every new pod. This one persists the state, so the
    fleet fails fast together and recovers together.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        pool: asyncpg.Pool,
        name: str = "llm",
        threshold: int = 5,
        reset_after: float = 30.0,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._pool = pool
        self.name = name
        self.threshold = threshold
        self.reset_after = reset_after

    async def state(self) -> str:
        """Current breaker state, resolving an elapsed cooldown to half-open."""
        row = await self._pool.fetchrow(
            "SELECT state, opened_at,"
            " EXTRACT(EPOCH FROM (now() - opened_at)) AS since_open"
            " FROM circuit_breakers WHERE name = $1",
            self.name,
        )
        if row is None or row["state"] != self.OPEN:
            return str(row["state"]) if row else self.CLOSED
        if row["since_open"] is not None and float(row["since_open"]) >= self.reset_after:
            return self.HALF_OPEN
        return self.OPEN

    async def before_call(self) -> None:
        """Raise `CircuitOpen` while the breaker is open and cooling down."""
        current = await self.state()
        if current == self.OPEN:
            row = await self._pool.fetchrow(
                "SELECT failures, EXTRACT(EPOCH FROM (now() - opened_at)) AS since_open"
                " FROM circuit_breakers WHERE name = $1",
                self.name,
            )
            remaining = self.reset_after - float(row["since_open"] or 0)
            raise CircuitOpen(
                f"circuit {self.name!r} open after {row['failures']} consecutive "
                f"failures; retry in {remaining:.1f}s"
            )

    async def record_success(self) -> None:
        """Close the breaker and clear the failure count."""
        await self._pool.execute(
            """
            INSERT INTO circuit_breakers (name, state, failures, opened_at, updated_at)
            VALUES ($1, 'closed', 0, NULL, now())
            ON CONFLICT (name) DO UPDATE
               SET state = 'closed', failures = 0, opened_at = NULL, updated_at = now()
            """,
            self.name,
        )

    async def record_failure(self) -> None:
        """Count a failure, tripping the breaker open at the threshold.

        The increment and the threshold check happen in one statement under a row lock,
        so concurrent workers cannot each read the same count and skip past the trip.
        """
        await self._pool.execute(
            """
            INSERT INTO circuit_breakers (name, state, failures, opened_at, updated_at)
            VALUES ($1, CASE WHEN 1 >= $2 THEN 'open' ELSE 'closed' END, 1,
                    CASE WHEN 1 >= $2 THEN now() ELSE NULL END, now())
            ON CONFLICT (name) DO UPDATE
               SET failures = circuit_breakers.failures + 1,
                   state = CASE
                       WHEN circuit_breakers.failures + 1 >= $2 THEN 'open'
                       ELSE circuit_breakers.state
                   END,
                   opened_at = CASE
                       WHEN circuit_breakers.failures + 1 >= $2 THEN now()
                       ELSE circuit_breakers.opened_at
                   END,
                   updated_at = now()
            """,
            self.name,
            self.threshold,
        )


class DistributedGuardedLLMClient:
    """`GuardedLLMClient` backed by Postgres-shared limiter and breaker state.

    Same contract as the in-process version, but the guarantees hold across restarts
    and across workers.
    """

    def __init__(
        self,
        client: object,
        limiter: DistributedRateLimiter | None = None,
        breaker: DistributedCircuitBreaker | None = None,
    ) -> None:
        self._client = client
        self.limiter = limiter
        self.breaker = breaker

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        """Rate-limit, check the shared breaker, then delegate."""
        if self.breaker is not None:
            await self.breaker.before_call()
        if self.limiter is not None:
            await self.limiter.acquire()
        try:
            completion: Completion = await self._client.complete(messages, tools)  # type: ignore[attr-defined]
        except Exception:
            if self.breaker is not None:
                await self.breaker.record_failure()
            raise
        if self.breaker is not None:
            await self.breaker.record_success()
        return completion
