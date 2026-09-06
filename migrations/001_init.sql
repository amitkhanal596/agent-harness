CREATE TABLE IF NOT EXISTS agent_runs (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task           TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'running',
    final_output   TEXT,
    step_count     INT  NOT NULL DEFAULT 0,
    total_tokens   INT  NOT NULL DEFAULT 0,
    total_cost_usd NUMERIC(10,6) NOT NULL DEFAULT 0,
    error          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS run_checkpoints (
    id          BIGSERIAL PRIMARY KEY,
    run_id      UUID NOT NULL REFERENCES agent_runs(id),
    step_number INT  NOT NULL,
    state_json  JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, step_number)
);

CREATE TABLE IF NOT EXISTS run_events (
    id            BIGSERIAL PRIMARY KEY,
    run_id        UUID NOT NULL REFERENCES agent_runs(id),
    step_number   INT  NOT NULL,
    kind          TEXT NOT NULL,          -- llm_call | tool_call
    tool_name     TEXT,
    input_json    JSONB,
    output_json   JSONB,
    latency_ms    INT,
    prompt_tokens INT,
    output_tokens INT,
    cost_usd      NUMERIC(10,6),
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON run_checkpoints(run_id, step_number);
CREATE INDEX IF NOT EXISTS idx_events_run      ON run_events(run_id, step_number);
