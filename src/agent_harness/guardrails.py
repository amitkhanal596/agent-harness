"""Rate limiting, circuit breaking, and the per-run budget check."""

from __future__ import annotations

import asyncio
import time

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
