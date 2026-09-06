-- §2.7: detect drift between trace events and the checkpoints they describe.
--
-- The loop writes a trace event and a checkpoint in separate transactions, so a crash
-- between them can leave an event whose step was never checkpointed. Making the two
-- writes one transaction would couple observability to durability — a tracer outage
-- would then fail the run — so instead the drift is made *detectable* and *reportable*.
--
-- This view lists events whose (run_id, step_number) has no corresponding checkpoint.
-- Under normal operation it is empty; a non-empty result means a run died mid-step.

CREATE OR REPLACE VIEW orphaned_events AS
SELECT e.run_id,
       e.step_number,
       count(*)          AS event_count,
       min(e.created_at) AS first_event_at,
       max(e.created_at) AS last_event_at
  FROM run_events e
 WHERE NOT EXISTS (
           SELECT 1
             FROM run_checkpoints c
            WHERE c.run_id = e.run_id
              AND c.step_number = e.step_number
       )
 GROUP BY e.run_id, e.step_number;

-- Runs left in a non-terminal state with no recent activity: the signature of a process
-- that died without persisting a terminal status. `agent doctor` surfaces these.
CREATE OR REPLACE VIEW stale_runs AS
SELECT id,
       task,
       status,
       step_count,
       total_cost_usd,
       updated_at,
       now() - updated_at AS idle_for
  FROM agent_runs
 WHERE status IN ('running', 'paused');
