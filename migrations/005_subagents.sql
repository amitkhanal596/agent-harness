-- §3.5: parent/child links between runs, so a delegated sub-task is a first-class run
-- with its own checkpoints, trace events, and cost — not an opaque blob inside its
-- parent's transcript.
--
-- Self-referencing FK on agent_runs: a child run is an ordinary run that happens to
-- know who spawned it, so every existing query, view, and CLI command works on it
-- unchanged.

ALTER TABLE agent_runs
    ADD COLUMN IF NOT EXISTS parent_run_id UUID REFERENCES agent_runs(id);

CREATE INDEX IF NOT EXISTS idx_runs_parent ON agent_runs(parent_run_id);

-- Roll-up of a run plus everything it delegated, for true total cost.
CREATE OR REPLACE VIEW run_trees AS
WITH RECURSIVE tree AS (
    SELECT id AS root_id, id, parent_run_id, total_cost_usd, total_tokens, 0 AS depth
      FROM agent_runs
     WHERE parent_run_id IS NULL
    UNION ALL
    SELECT t.root_id, r.id, r.parent_run_id, r.total_cost_usd, r.total_tokens, t.depth + 1
      FROM agent_runs r
      JOIN tree t ON r.parent_run_id = t.id
)
SELECT root_id,
       count(*)                AS run_count,
       max(depth)              AS max_depth,
       sum(total_cost_usd)     AS tree_cost_usd,
       sum(total_tokens)       AS tree_tokens
  FROM tree
 GROUP BY root_id;
