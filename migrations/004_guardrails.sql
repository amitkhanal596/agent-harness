-- §2.2: cross-process guardrail state.
--
-- The in-process RateLimiter and CircuitBreaker are correct for a single worker but
-- guarantee nothing beyond it: a restart resets an open breaker, and N workers each get
-- their own token bucket, so the real aggregate rate is N times the configured one.
--
-- Postgres is already the durability store, so the shared state lives here rather than
-- adding Redis. Both tables are tiny and hot, updated with row-level locks.

-- One row per named breaker. `failures` and `state` survive restarts, so a provider
-- outage keeps failing fast across deploys instead of being retried by every new process.
CREATE TABLE IF NOT EXISTS circuit_breakers (
    name        TEXT PRIMARY KEY,
    state       TEXT        NOT NULL DEFAULT 'closed',   -- closed | open | half_open
    failures    INT         NOT NULL DEFAULT 0,
    opened_at   TIMESTAMPTZ,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per named bucket, storing the token count and when it was last refilled.
-- Refill is computed from the elapsed time on read, so no background job is needed.
CREATE TABLE IF NOT EXISTS rate_limit_buckets (
    name        TEXT PRIMARY KEY,
    tokens      DOUBLE PRECISION NOT NULL,
    updated_at  TIMESTAMPTZ      NOT NULL DEFAULT now()
);
