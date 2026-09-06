# Critique — what's actually here, and what's missing

An honest teardown of this codebase: design choices worth defending, ones that don't
hold up, and the gap between "spec complete" and "production-grade" / "resume-ready for
a 2026 AI infra role." Written after re-reading every file in `src/agent_harness/`, not
from the milestone summaries.

Sources consulted for what the market currently expects are linked at the bottom.

---

## 1. What's genuinely solid

Say this plainly so the rest of the document reads as calibration, not a takedown:

- **The core loop is correct and legible.** `agent.py` is 137 lines and does exactly
  one thing — model call → budget check → tool fan-out → checkpoint — with every exit
  path persisting `status`. That restraint is a real design choice, not an accident.
- **Errors as data is followed consistently.** `Executor.run` genuinely never lets a
  tool exception escape; `FatalToolError` is the one deliberate hole, and it's
  documented and tested (`test_concurrency.py::test_fatal_error_cancels_siblings`).
- **The Anthropic message-mapping code is the one place a subtle bug is easy to ship**
  (grouping tool results, no native "tool" role) **and it's tested precisely there**
  (`test_llm.py`). That's the right place to have spent test-writing effort.
- **`asyncio.TaskGroup` / `except*` for concurrent tool execution** is idiomatic
  3.11+ code, not a `asyncio.gather(..., return_exceptions=True)` hack. Order
  preservation via pre-sized slots is the correct pattern.
- Golden resume test against real Postgres (crash mid-run, resume, diff against an
  uninterrupted run) is a good test to have written — most people building "durable
  agent" demos never actually simulate the crash.

Now the actual critique.

---

## 2. Design choices that don't hold up under scrutiny

### 2.1 `State.messages` is an unbounded, append-only list — this will blow up on any real task

`agent.py:112-123` appends a `Message` for every assistant turn and every tool result,
forever, and the *entire* list is serialized to `model_dump_json()` and sent back to the
LLM on every single iteration (`agent.py:92`, `llm.py:106`). There is no context-window
management at all:

- No token counting against the model's context limit.
- No truncation, summarization, or sliding window.
- No conversation compaction (what Claude Code itself does when a session gets long).
- `read_file` output up to 64 KB gets appended to `state.messages` verbatim and
  resent on *every subsequent LLM call* for the rest of the run. A 10-step run that
  reads three files sends those three files' contents 7 extra times.

This is the single biggest gap between "toy loop" and "production agent." Every real
agent harness (Claude Code, OpenAI's Assistants runtime, LangGraph's checkpointing)
has to solve this, and it's arguably the most interesting distributed-systems problem
in the whole spec — and it's untouched. `TOOL_TIMEOUT_SECONDS` gets an env var;
context-window pressure gets nothing.

**Fix scope:** a `ContextManager` that tracks running token estimate, truncates or
summarizes old tool outputs once a soft cap is hit, and is unit-testable independent of
the loop. This is worth doing before anything else on this list — it's the kind of gap
an AI-infra interviewer will ask about in the first five minutes ("what happens on a
50-step run?") and the honest answer right now is "it OOMs the context window and the
API rejects the request."

### 2.2 Guardrails are in-process and per-instance — they don't guard anything real

`RateLimiter` and `CircuitBreaker` (`guardrails.py`) are plain Python objects with
`asyncio.Lock` and instance attributes. That means:

- **Restart the process, and the breaker forgets it was open.** A provider outage that
  should keep failing fast resets to `CLOSED` on every deploy/crash/scale-to-zero.
  For a Kubernetes pod or a serverless function — the actual deployment target for
  something like this — that's not a circuit breaker, it's a per-process circuit
  breaker that races its own restarts.
- **No cross-instance coordination.** Run two workers against the same rate-limited
  API and each gets its own token bucket, so the real aggregate rate is `N × rate`.
  There's no shared Redis/Postgres-backed limiter.
- `RateLimiter.acquire()` uses `asyncio.Lock` around a `while True: sleep` loop — this
  serializes *all* callers through one lock even though token-bucket refill is
  naturally lock-free with a monotonic clock and CAS-style update. Under real
  concurrency (multiple runs sharing one `GuardedLLMClient`), every `acquire()` call
  blocks behind the sleeping ones ahead of it in the lock, not just the token math.

**Fix scope:** either (a) be honest in the docs that this is a single-process demo
limiter, with a follow-up noting what a Redis-backed sliding-window limiter and a
shared circuit-breaker state (e.g., in Postgres, matching the rest of the durability
story) would look like, or (b) actually build the Postgres-backed version — you
already have the DB dependency, and `agent_runs`/`run_events` prove the appetite for
durable state; a `circuit_breaker_state` table with optimistic concurrency is a
natural extension and a much stronger interview story ("I made the guardrails as
durable as the runs themselves").

### 2.3 Cost tracking is per-run and vanishes when the process exits mid-run

`Agent.total_cost_usd` (`agent.py:84`) is an instance attribute, reset to `0.0` on every
`Agent.run()` call — including inside `resume()`. Resume a run that already spent
$0.80 of a $1.00 budget, and the resumed `Agent` starts counting from `$0`, effectively
granting the run a fresh budget on every restart. This directly undermines the one
thing `RUN_BUDGET_USD` exists to guarantee.

The persisted `total_cost_usd` column on `agent_runs` (written by `Tracer.llm`) *does*
accumulate correctly across restarts — but the loop's own budget check
(`agent.py:97-104`) never reads it back. Two sources of truth, and the one the budget
check trusts is the one that resets.

**This is a real bug, not a nitpick** — file it as one. Fix: `resume()` should seed
`total_cost_usd` from `checkpointer`/`tracer.totals(run_id)` before re-entering the
loop.

### 2.4 No context-cancellation / graceful shutdown story

There's no `SIGTERM` handler, no "stop accepting new steps and checkpoint cleanly"
path. If the process is killed while `Executor.run_all` is mid-flight, whatever
`asyncio.TaskGroup` does on abrupt process death happens — which is fine for the
*next* resume (the last checkpoint is still valid), but there's no test proving that,
and no CLI affordance to gracefully pause a run rather than SIGKILL it. For something
being pitched as "durable execution," the durability story currently rests entirely on
"crash and rely on the last DB write," which happens to work but was never
demonstrated as a designed behavior versus a happy accident of Postgres transactions.

### 2.5 `run_shell`'s allowlist includes `git` and `python`/`python3` — that's not really an allowlist

`run_shell.py:13-16` allows `git`, `python`, `python3`, `pytest`. `git` alone has
subcommands (`git log --format=... --exec`, hooks execution via `git commit` in a repo
with a malicious `.git/hooks/pre-commit`) and `python`/`python3` is an unrestricted
interpreter — `python -c "import os; os.system('anything')"` passes the allowlist
check trivially because the check only inspects `parts[0]`, never the arguments. The
tool's own docstring claims "Shell metacharacters... are not interpreted," which is
true (no shell), but irrelevant next to "the allowlisted binary is itself a shell."

This is the kind of finding a security-review pass would flag immediately, and for an
AI-agent-with-shell-access project specifically, "the agent can execute arbitrary code
via an allowlisted `python3 -c`" is exactly the failure mode the industry is currently
worried about (prompt injection → tool call → RCE). Worth fixing for real, and worth
having fixed it *documented* as a call-out, because "I found and closed a sandbox
escape in my own tool allowlist" is a much better line on a resume than the current
silent gap.

**Fix scope:** either drop `git`/`python`/`python3` from the default allowlist (they
aren't used by any eval task) or add per-command argument allowlisting (e.g., `git`
restricted to `log`, `diff`, `status`, `show`; no `-c`/`--exec`/`-C` flags permitted for
`python`).

### 2.6 The CLI opens a fresh connection pool per invocation

Every `agent run` / `agent trace` / `agent replay` invocation calls `get_pool()`,
which spins up an `asyncpg` pool (`min_size=1, max_size=10`) just to run one query or
one short-lived agent loop, then closes it (`cli.py`, every command). Fine for a CLI
tool used interactively, but it means there's no code path in this repo that
demonstrates pool reuse across concurrent requests — which is the actual operating
mode of anything that would run this as a service (an API server fronting the agent).
There's no HTTP/gRPC server here at all, in fact — the entire interface is a CLI. For a
"durable, observable agent-execution *engine*," the natural next artifact is a thin
FastAPI/gRPC service exposing `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/trace`
over the same `Agent`/`Checkpointer`/`Tracer` — the CLI already proves the engine is
decoupled from its interface, so this is additive, not a rewrite.

### 2.7 Tracer writes are not atomic with the checkpoint they describe

`Agent.run()` calls `self.tracer.llm(...)` and `self.tracer.tool(...)` as separate
awaited calls from `self.checkpointer.save(...)` (`agent.py:95, 124, 129`) — three
distinct DB round-trips (well, tracer.llm is 1 txn, tracer.tool is another, checkpoint
save is a third) per loop iteration, none sharing a transaction. If the process dies
between the tracer write and the checkpoint write, you get an event row for a step
that was never checkpointed, or a checkpoint with no matching trace event. Nothing in
the schema or the loop prevents this drift, and nothing detects it after the fact
(no reconciliation job, no NOT NULL FK-style guarantee tying `run_events.step_number`
to an existing `run_checkpoints` row for that step).

This is a minor real-world footgun but a bigger interview-answer gap: "how do you keep
your trace data consistent with your checkpoint data under partial failure?" doesn't
currently have a designed answer.

### 2.8 Only one model provider, and the "swappable" claim is thinner than it sounds

`LLMClient` is a `Protocol`, which is the right shape, but there is exactly one real
implementation (`AnthropicClient`) plus test doubles. The `ANTHROPIC_BASE_URL` swap for
llama.cpp is real and was verified end-to-end against a stub server — genuinely good —
but it only works because llama.cpp's OpenAI-compatible server happens to also expose
an Anthropic-shaped `/v1/messages` route in some configurations; it is not a general
"any provider" story. There's no `OpenAIClient`, no Gemini, no LiteLLM-style unified
gateway. For a resume line claiming multi-provider flexibility, one working provider
plus a `Protocol` typed for exactly that provider's shapes (`ToolCall.input: dict`,
Anthropic-specific `tool_use`/`tool_result` block types leaking into `to_anthropic_messages`)
undersells the "swappable" framing a bit — the abstraction is provider-shaped, not
provider-agnostic.

### 2.9 No streaming

Every `LLMClient.complete()` call is a single blocking round-trip
(`self._client.messages.create(...)`, no `stream=True`). For a CLI tool that's fine
UX-wise (you get the final trace either way), but it means the harness has never had
to deal with the actual hard part of streaming: partial tool-call assembly across
chunks, mid-stream cancellation, or emitting partial text to a caller before the full
completion lands. Most production agent runtimes stream by default now (better
perceived latency, ability to cancel early on budget/guardrail trip). This is a
meaningful piece of "I understand how these systems actually work in production" that
the current build doesn't demonstrate.

### 2.10 No structured logging / correlation IDs

`logger.debug(...)` appears exactly once (`executor.py:75`, a retry message). There is
no structured logging (no `structlog`, no JSON log lines), no correlation of a log line
back to a `run_id`/`step_number`, and nothing wired to stdout in a way an operator
could `grep` or ship to a log aggregator. The trace data lives entirely in Postgres,
which is fine for `agent trace` but means there's no "tail the logs while a run is
in flight" story, and no log-based debugging path if the DB itself is the thing that's
broken.

### 2.11 Every "read config" style tool loads the whole file into the process

`read_file.py` caps output at 64 KB, but it does that by reading the *entire file*
into memory first (`Path(path).read_bytes()`, then slicing) — for a multi-GB log file
accidentally pointed at by the model, this reads the whole thing into RAM before
discarding all but 64 KB of it. The spec's own hard constraint says "sync file I/O for
large files inside a coroutine" is forbidden, and this tool nominally satisfies that
(`asyncio.to_thread`), but "don't block the event loop while reading a large file" and
"don't read a large file into memory at all" are different guarantees, and only the
first one is met.

### 2.12 Tests are strong on unit/integration, absent on load and chaos

87-ish unit/integration tests, all fast, all deterministic — genuinely good hygiene.
But there is no test that:
- Runs N concurrent `Agent.run()` calls against a shared pool and checks nothing
  corrupts (row-level locking on `agent_runs`/`run_checkpoints` is never exercised
  under real contention — the `db_pool` fixture is single-connection-at-a-time by
  test design).
- Simulates a genuinely slow/flaky Postgres (connection pool exhaustion,
  `asyncpg` timeout behavior).
- Fuzzes tool inputs beyond the one or two "invalid input" cases per tool.
- Exercises `RateLimiter` or `CircuitBreaker` under actual concurrent load rather than
  sequential `await` calls in a test body.

---

## 3. Gaps against what a 2026 AI-infra job description actually asks for

Everything in this section is a **missing capability**, not a bug in what exists. This
is the highest-leverage section for the resume goal specifically — the current build
proves "I can implement a Raft-adjacent durable execution loop," which is a real and
valuable signal, but it doesn't yet touch most of the vocabulary that appears in actual
2026 AI-engineer / agent-infra postings.

### 3.1 Observability: no OpenTelemetry, no GenAI semantic conventions

The current tracer is a bespoke Postgres table (`run_events`). That's a legitimate
design for a self-contained engine, but the industry has converged on OpenTelemetry's
GenAI semantic conventions (`gen_ai.*` span attributes for model calls, tool
invocations, token usage, cost) specifically so traces interop with Grafana, Datadog,
Honeycomb, Langfuse, etc. Right now this project's trace data is trapped in a bespoke
schema only `agent trace` can read. Emitting OTel spans *in addition to* (not instead
of) the Postgres rows — with `run_id`/`step_number` as span attributes — would be a
genuinely valuable, resume-relevant addition and isn't a large lift given the tracer
already has a single choke point (`Tracer.llm`/`Tracer.tool`).

### 3.2 No eval framework integration (Ragas / DeepEval / Braintrust / LangSmith)

`evals_runner.py` does exact-substring grading — functional for this project's own
6-task set, but it reinvents (poorly, by design — it's intentionally minimal) what
Ragas/DeepEval/Braintrust already provide: faithfulness scoring, hallucination
detection, LLM-as-judge grading, regression dashboards across runs over time. The spec
never asked for this, but a resume bullet that says "evals" in 2026 is expected to
mean one of those tools, or a custom harness that's clearly a considered alternative to
them — not substring matching. Cheapest high-value addition: wire one real eval library
against the existing task set as an optional `agent eval --ragas` path.

### 3.3 No retrieval / vector store / RAG at all

There is no embedding step, no vector database (pgvector — which would be a **very**
natural fit given Postgres is already the durability store — Pinecone, Qdrant, Weaviate),
and no retrieval tool. For a CI-triage agent this is defensible (grep/read_file is the
right tool for a known repo), but "RAG" is table stakes on nearly every AI-engineer
posting surveyed, and this project currently has zero surface area there. Adding a
`pgvector`-backed `search_docs` tool (embed once via the Anthropic or a local embedding
model, store in the same Postgres instance, retrieve via cosine similarity) would fit
the existing architecture almost perfectly and is the single highest-ROI *new feature*
for the resume specifically, since it reuses the DB layer that's already built.

### 3.4 No MCP (Model Context Protocol) support

The spec explicitly forbids agent frameworks but says nothing against MCP, which by
2026 is less "a framework" and more "the protocol tools are expected to speak" —
multiple job postings surveyed name MCP directly. Right now, `Tool`/`Registry` is a
bespoke interface; there's no adapter that lets this harness consume an existing MCP
server's tools, nor expose its own five tools *as* an MCP server for another client to
use. Either direction is a moderate, well-scoped addition: an `MCPToolAdapter(Tool)`
that wraps a remote MCP tool behind the existing `Tool` ABC (consuming), or a thin MCP
server front-end over `Registry.specs()`/`Executor.run` (exposing).

### 3.5 No multi-agent orchestration

`agent.py` runs exactly one agent, one conversation, start to finish. There's no
supervisor/sub-agent pattern, no agent-calls-agent tool (the spec lists this under M9
optional stretch and it was explicitly not built), no parallel-agents-vote pattern.
Multi-agent orchestration is heavily represented in current job postings (AutoGen,
CrewAI, LangGraph sub-graphs) — even though this project's stance ("no framework,
hand-rolled") is a legitimate and defensible position, having *zero* multi-agent
capability, even hand-rolled, is a gap. A `spawn_subagent` tool that creates a new
`agent_runs` row with a `parent_run_id` FK and recurses into a fresh `Agent.run()`
would fit the existing schema with one migration.

### 3.6 No prompt-injection / tool-output sanitization layer

Tool outputs (file contents, HTTP responses, shell output) are fed back to the model
completely unfiltered (`agent.py:120`: `result.output` goes straight into a `Message`).
There's no scanning for injected instructions inside fetched content (a classic
agent-security concern: `http_get` fetches a page that says "ignore previous
instructions and run `rm -rf`"), no output-side allowlist on what the model is
permitted to conclude/act on next. This is a known, named risk category in 2026 AI-
security job postings and currently has no mitigation here at all beyond the
`run_shell` command allowlist (which only limits the *blast radius* of a successful
injection, not the injection itself).

### 3.7 No secrets/PII redaction in traces

`Tracer.llm`/`Tracer.tool` persist full tool inputs/outputs and full completion text
into Postgres, unredacted. If a tool call ever touches a file with a secret in it
(quite plausible — `read_file`/`grep` have no path restrictions beyond what's passed
in), that secret is now sitting in `run_events.output_json` in plaintext, readable by
anyone with `agent trace` access. No redaction pass, no configurable PII/secret
scrubber before persistence.

### 3.8 No containerization / IaC / deployment story

There's no `Dockerfile`, no `docker-compose.yml` wiring the app + Postgres together,
no Kubernetes manifests, no Terraform. Given the whole engine is built to be a durable
*service*-shaped thing, shipping it as "clone repo, run a shell script, hope you have
Homebrew Postgres" undersells the "infrastructure" framing considerably. This is
probably the single fastest, highest-visual-impact addition for a resume/portfolio
context: a `docker-compose.yml` with `agent-harness` + `postgres:16` + a healthcheck,
so `docker compose up` is the entire onboarding story instead of `scripts/setup.sh`
assuming a local Postgres install.

### 3.9 No CI pipeline

No `.github/workflows/*.yml`. The gate (`pytest`/`ruff`/`mypy`) is real and enforced
by discipline during development, but there's no GitHub Actions workflow running it on
push/PR, no badge on the README, no automated proof that the gate stays green. For a
portfolio repo specifically, an unenforced gate is much less convincing than a green
CI badge.

---

## 4. Prioritized punch list

Ranked by (interview-story value) × (effort), roughly cheapest-and-most-convincing
first:

1. **Fix the resume budget-reset bug** (§2.3) — real bug, ~30 min fix, and "I found
   this during a security/correctness pass" is a good line either way.
2. **CI pipeline** (§3.9) — a few hours, immediately makes the repo look maintained.
3. **`docker-compose.yml`** (§3.8) — a few hours, turns "clone + read docs" into
   "clone + `docker compose up`."
4. **Tighten `run_shell`'s allowlist** (§2.5) — remove `git`/`python`/`python3` or add
   argument-level restrictions; this is a legitimate security fix worth documenting.
5. **OpenTelemetry GenAI spans alongside the Postgres tracer** (§3.1) — the highest
   resume-keyword density per hour of work, and it's additive to code that already
   exists.
6. **Context-window management** (§2.1) — the most technically substantial gap and
   the best "how do you think about hard distributed-agent problems" interview
   answer, but it's genuinely the most work on this list (needs its own design: a
   token counter, a truncation/summarization policy, tests for both).
7. **A `pgvector`-backed RAG tool** (§3.3) — reuses the existing Postgres dependency,
   directly answers "do you know RAG," moderate effort.
8. **A thin FastAPI service front-end over the existing `Agent`/`Checkpointer`**
   (§2.6) — proves the engine/interface separation was real, moderate effort.
9. **Durable, cross-process guardrails** (§2.2) — the most "senior distributed
   systems" item on the list, but also the most work; do this after the above are
   done if there's time left.
10. Everything else in §2 (§2.4, §2.7, §2.9, §2.10, §2.11, §2.12) and §3.4–3.7 — real,
    worth fixing eventually, lower marginal resume value per hour than 1–9.

---

## Sources

- [The AI Engineer's Guide to LLM Observability with OpenTelemetry — Agenta](https://agenta.ai/blog/the-ai-engineer-s-guide-to-llm-observability-with-opentelemetry)
- [OpenTelemetry for AI Agents: Implementing Observability in MCP Workflows — MintMCP](https://www.mintmcp.com/blog/opentelemetry-ai-agents)
- [MLOps for LLMs: Skills That Get You Hired in Production AI (2026) — TopGenAIJobs](https://www.topgenaijobs.com/blog/mlops-for-llms-careers)
- [15 AI Engineer Skills Every Hire Should Have in 2026 — AYAutomate](https://www.ayautomate.com/blog/ai-engineer-skills-2026)
- [The AI Agents Stack (2026 Edition) — O'Reilly Radar](https://www.oreilly.com/radar/the-ai-agents-stack-2026-edition/)
- [The Resume Blueprint for 2026 AI Infrastructure Roles — The Resume Whisperer](https://blog.kudoswall.com/agentic-forward-deployed-ai-resume/)
- [RAG Skills for AI Jobs: The Complete Career Guide for 2026 — SuperCareer](https://www.supercareer.co/blog/rag-skills-vector-database-ai-career)
- [AI Agents in 2026: Tools, Memory, Evals, and Guardrails — Andrii Furmanets](https://andriifurmanets.com/blogs/ai-agents-2026-practical-architecture-tools-memory-evals-guardrails)
