# Design decisions — where I diverged from CRITIQUE.md

Every item in `CRITIQUE.md` was addressed. In five places I did something *other* than
what the critique literally recommended, because the recommendation would have made the
system worse. Those are documented here with the reasoning, so the divergence is a
decision rather than an oversight.

---

## 1. §2.7 — I did **not** make tracer writes atomic with checkpoints

**The critique asked for:** trace events and checkpoints in one transaction, so a crash
cannot leave an event whose step was never checkpointed.

**What I did instead:** kept them in separate transactions and made the resulting drift
*detectable* — `orphaned_events` / `stale_runs` views, `Tracer.orphaned_events()`, and an
`agent doctor` command that reports both and exits non-zero.

**Why.** Sharing a transaction couples observability to durability in the wrong
direction: a tracer failure would then roll back the checkpoint and fail the run. That
trade is backwards. Losing a metrics row costs a gap in a dashboard; losing a checkpoint
costs the entire run, including everything already paid for in model tokens. Observability
should degrade independently of the thing it observes.

There is also a limit to what one transaction would buy. It closes the window between the
tracer write and the checkpoint write, but not the window between the *model call* and
either write — a crash there still leaves spend that no row records. The honest framing is
"these are separate writes and drift is possible", with tooling to find it, rather than a
guarantee that only appears complete.

**What would change my mind:** if trace data were used for billing rather than
diagnostics, the correctness requirement would flip and atomicity would be worth the
coupling.

---

## 2. §3.2 — I did **not** take a Ragas / DeepEval / Braintrust dependency

**The critique asked for:** wiring in one of the standard eval libraries, noting that
"a resume bullet that says evals in 2026 is expected to mean one of those tools."

**What I did instead:** extended the built-in grader with regex criteria, numeric
matching with tolerance, LLM-as-judge rubrics, per-criterion failure reporting, and
partial-credit scoring.

**Why.** Three reasons, in order of weight:

1. **The spec forbids it.** `README.md` §1 is explicit: "Add abstractions, layers, config
   options, or dependencies not specified here" is a MUST NOT, and the eval requirement
   in M8 is `evals/tasks.jsonl` plus three aggregate metrics. Adding a framework to
   decorate a resume, against a written constraint, is exactly the instinct the spec was
   written to suppress.
2. **Those tools solve a different problem.** Ragas is built for RAG-specific metrics
   (faithfulness, context precision) over a retrieval corpus. This harness's eval set
   grades *agent task completion* — did it find the bug, did it report the right port.
   The overlap is thin, and most of what Ragas provides would sit unused.
3. **The gap the critique identified was real, and I fixed it.** The actual weakness was
   not "no framework", it was that grading returned an opaque boolean. That is now fixed:
   a failing eval reports *which criterion* failed and how close the answer was. That is
   the capability worth having, and it is ~150 lines rather than a dependency.

**Note on the resume framing.** I think "built a grading harness with regex, numeric,
and LLM-judge criteria and per-criterion diagnostics" is a stronger interview answer than
"imported Ragas" — it demonstrates understanding of what eval frameworks *do*. If a job
description names a specific tool, the honest move is to learn that tool, not to
`pip install` it once for a portfolio repo.

---

## 3. §2.2 — Distributed guardrails are **opt-in**, not the default

**The critique asked for:** durable, cross-process guardrails, noting the in-process ones
"don't guard anything real."

**What I did instead:** built `DistributedRateLimiter` and `DistributedCircuitBreaker`
backed by Postgres, but left the in-process versions as the default behind
`DISTRIBUTED_GUARDRAILS=false`.

**Why.** The in-process versions are not wrong — they are correct for a single-worker
deployment, which is what `agent run` on a laptop actually is. The distributed versions
cost a database round-trip on **every** model call and every token claim. Making them the
default would tax the common case to fix a problem the common case does not have.

The critique's framing ("don't guard anything real") is right about a multi-worker
deployment and wrong about a single process. Both are now available and the docstrings
say which is appropriate when.

**Also worth saying:** I used Postgres rather than Redis, which the critique suggested as
an option. Postgres is already a hard dependency; Redis would be a new one, for state
that is small, low-throughput, and benefits from the transactional semantics Postgres
already provides.

---

## 4. §2.9 — Streaming is a **separate Protocol**, not the default path

**The critique asked for:** streaming, noting "most production agent runtimes stream by
default now."

**What I did instead:** added `StreamingLLMClient` as a Protocol extending `LLMClient`,
with `AnthropicClient.stream()`. The loop still uses `complete()`.

**Why.** The loop cannot act on a partial response. It needs a *finished* `Completion`
before it can validate arguments and execute tools — acting on a half-assembled
`tool_use` block would be a correctness bug, not a latency win. Streaming's benefit is
perceived latency for a human watching output, which is a property of the *interface*,
not the engine.

Making streaming the loop's default would mean assembling the full completion anyway and
then pretending it arrived incrementally. The narrower Protocol lets a UI consume deltas
while the loop keeps the simpler contract. This is the same reasoning as §2.6: the engine
should not absorb concerns that belong to the interface.

---

## 5. §2.6 — The API is synchronous; I did **not** add a job queue

**The critique asked for:** "a thin FastAPI service front-end over the existing
`Agent`/`Checkpointer`."

**What I did:** exactly that — but `POST /runs` runs to completion and returns the
answer, rather than returning a job id immediately.

**Why.** A job queue is a materially larger design decision than "add an HTTP interface":
it needs a worker pool, a queue backend, job leasing, retry/visibility semantics, and a
story for workers dying mid-job. That is a second system, and the critique asked for a
thin front-end.

The pieces that *would* make async work already exist — runs are durable, resumable, and
queryable, so a worker could pick up `status='running'` rows and `agent doctor` already
finds abandoned ones. The endpoint docstring says the synchronous choice is deliberate
and points at polling `GET /runs/{id}` as the pattern for long tasks, rather than leaving
a reader to assume nobody thought about it.

---

## Things the critique said that I simply agreed with

For balance — the large majority of the critique was correct and implemented as written,
including the two items that turned out to be genuine bugs:

- **§2.3 the resume budget reset** was a real bug that voided the budget guarantee across
  restarts. Fixed, with a test verified to fail against the old code.
- **§2.5 the `run_shell` allowlist** was a real sandbox escape — `python3 -c` passed an
  argv[0]-only check trivially. Fixed by removing interpreters and validating arguments.
- **§2.1 the unbounded context window** was the most substantial gap and needed the
  design work the critique implied.
- **§2.11 `read_file` slurping whole files** was real, and the 2 GiB sparse-file test
  fails against the old implementation.

Two further bugs surfaced while implementing, neither of which the critique predicted:

- **`MIGRATIONS_DIR` resolved to a nonexistent path once installed** (found by
  containerizing) — migrations silently applied nothing inside Docker.
- **NUL bytes in tool output killed runs** (found by the §2.12 fuzzer) — Postgres `jsonb`
  cannot store ` `, so reading any binary file destroyed the run.

Both are documented in `PROGRESS.md` with their fixes and regression tests.
