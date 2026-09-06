# Agent Harness — Operator's Guide

A durable, observable agent-execution engine: an LLM loop with tools that checkpoints
every step to Postgres (so runs resume and replay), executes tools concurrently, and
traces cost, latency, and failures. No agent framework — the loop, executor,
checkpointer, and concurrency are hand-written; the only model dependency is the raw
`anthropic` SDK, behind a `Protocol` so it is swappable.

The build spec is `README.md`. Milestone-by-milestone status is `PROGRESS.md`.

---

## Setup

Everything below is also in `scripts/setup.sh`, which is executable and idempotent:

```bash
./scripts/setup.sh          # venv + install + Postgres check + migrations
```

Done by hand:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Postgres — either Docker:
docker run --name agent-pg -e POSTGRES_PASSWORD=dev -p 5432:5432 -d postgres:15
# ...or a local server (Homebrew), creating the role and database the DSN expects:
createdb agent
psql -d agent -c "CREATE ROLE postgres LOGIN SUPERUSER PASSWORD 'dev';"

cp .env.example .env        # then fill in ANTHROPIC_API_KEY
agent migrate               # apply migrations/*.sql (safe to re-run)
agent run "read ./config.txt and tell me the configured port"
```

`.env` values may carry trailing `# comments` — they are stripped, except for a `#`
inside a token such as a password or URL fragment.

---

## Commands

| Command | What it does |
|---|---|
| `agent run "<task>"` | Start a run; prints the run id up front, then the final answer. |
| `agent resume <run_id>` | Reload the newest checkpoint and continue the loop. |
| `agent replay <run_id>` | Print every checkpoint in order: step number, message count, status. |
| `agent trace <run_id>` | Per-event latency / tokens / cost, plus run totals. |
| `agent eval [--mock] [--limit N]` | Run `evals/tasks.jsonl` headlessly; prints pass rate, avg cost, avg latency. |
| `agent index <paths>` | Embed files into the vector store for `search_docs`. |
| `agent doctor` | Report orphaned trace events and stalled runs; non-zero on findings. |
| `agent migrate` | Apply pending migrations. |
| `agent config` | Print effective settings, API key redacted. |
| `agent-api` | Run the HTTP service (see **Service** below). |

`agent eval --mock` runs the whole set offline against a scripted model, so the command
is demonstrable with no API key and no network.

---

## Running as a service

```bash
agent-api                      # or: docker compose up
curl localhost:8000/health
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
     -d '{"task": "read ./config.txt and tell me the port"}'
```

| Route | Purpose |
|---|---|
| `POST /runs` | Start a run, return its outcome. |
| `GET /runs/{id}` | Run status, steps, tokens, cost. |
| `GET /runs/{id}/trace` | Per-event timeline plus totals. |
| `GET /runs/{id}/checkpoints` | `agent replay` over HTTP. |
| `POST /runs/{id}/resume` | Continue from the newest checkpoint. |
| `GET /health` | Liveness plus a real database round-trip. |
| `GET /health/consistency` | `agent doctor` over HTTP. |

Unlike the CLI, the service holds **one connection pool for the process lifetime**,
shared across concurrent requests. `POST /runs` is synchronous by design — see
`DESIGN_DECISIONS.md` §5.

---

## Docker

```bash
docker compose up              # postgres (pgvector) + migrations + the agent
docker compose run --rm agent run "read ./config.txt and tell me the port"
```

The image runs as a non-root `agent` user, which matters here specifically because the
process executes model-chosen commands. Do **not** pass `--abort-on-container-exit`: it
tears down Postgres the moment the one-shot `migrate` service exits.

---

## Configuration

All settings come from the environment or `.env` (`src/agent_harness/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for real runs. |
| `DATABASE_URL` | `postgres://postgres:dev@localhost:5432/agent` | Postgres DSN. |
| `MODEL` | `claude-sonnet-4-5` | Model id. Confirm the current id before use. |
| `MAX_STEPS` | `25` | Iteration cap; exceeding it raises `MaxStepsExceeded`. |
| `TOOL_TIMEOUT_SECONDS` | `30` | Per-tool-call timeout. |
| `TOOL_MAX_RETRIES` | `2` | Retries for transient tool failures. |
| `RUN_BUDGET_USD` | `1.00` | Per-run spend cap; exceeding it raises `BudgetExceeded`. |
| `ANTHROPIC_BASE_URL` | — | Point at an OpenAI-compatible `llama.cpp` server for local runs. |
| `CACHE_PROMPT` | `false` | Mark tool definitions with a cache breakpoint (Anthropic prompt caching). |
| `MAX_CONTEXT_TOKENS` | `150000` | Compact the conversation above this estimated size. |
| `KEEP_RECENT_MESSAGES` | `8` | Recent messages never touched by compaction. |
| `OTEL_EXPORTER` | `none` | `none` / `console` / `otlp` — emit GenAI-convention spans. |
| `OTEL_ENDPOINT` | — | OTLP collector URL when `OTEL_EXPORTER=otlp`. |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `true` | Structured JSON logs on stderr, correlated by `run_id`. |
| `VOYAGE_API_KEY` | — | Enables real semantic embeddings; falls back to a local hash embedder. |
| `DISTRIBUTED_GUARDRAILS` | `false` | Share rate-limit/breaker state across workers via Postgres. |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Bind address for `agent-api`. |

Pricing constants live in `llm.py` (`PRICE_PER_1M_INPUT`, `PRICE_PER_1M_OUTPUT`) and are
documented, not fetched — verify them before trusting the cost numbers.

---

## How it works

```
Agent.run(state)
  └─ loop while step_count < max_steps
       ├─ llm.complete(messages, registry.specs())      ── traced: latency, tokens, cost
       ├─ budget check                                  ── over? status=failed, BudgetExceeded
       ├─ no tool calls? append answer, status=completed, checkpoint, return
       ├─ executor.run_all(tool_calls)                  ── concurrent, order-preserving
       ├─ append one tool message per result            ── errors become "ERROR: ..." text
       └─ step_count += 1; checkpoint                   ── every iteration
     cap reached → status=failed, MaxStepsExceeded
```

Design rules the code holds to:

- **A tool failure is data.** The executor converts every tool exception into
  `ToolResult.error`, which the loop feeds back to the model as the tool result. The
  only exceptions that escape the loop are `BudgetExceeded` and `MaxStepsExceeded`.
- **Validate before executing.** Every tool input is checked against the tool's
  `InputModel` before the tool runs; a schema violation is an error result, not a crash.
- **Checkpoint every iteration.** `run_checkpoints` is upserted on
  `(run_id, step_number)`, so a resumed run rewrites its step rather than colliding.
- **Status is always persisted**, on every exit path.

### Safety

Tool output is **untrusted input**. Before it reaches the model, the checkpoint, or the
trace tables, every tool result passes through one sanitization step that:

- **strips characters Postgres cannot store** (a NUL byte in a binary file used to fail
  the checkpoint write and destroy the run — always applied, even with redaction off);
- **redacts credentials** (provider keys, PEM blocks, JWTs, DSN passwords, named
  assignments), preserving useful shape — a DSN keeps its host and user, loses only the
  password;
- **wraps injection attempts** in an explicit untrusted-data envelope naming the tool,
  rather than stripping phrases, so the model is told the content is data and can still
  analyse it.

These are heuristic defence-in-depth behind the real controls: the `run_shell`
allowlist (which validates arguments, not just the program — no interpreters), the
non-root container user, and the per-run budget.

### Tools

| Tool | Behavior | Guardrail |
|---|---|---|
| `read_file(path)` | File contents | 64 KB cap, truncation noted |
| `list_dir(path)` | Entries, one per line | 500-entry cap |
| `grep(pattern, path)` | Numbered matching lines, file or tree | 200-match cap, 1 MB/file |
| `run_shell(command)` | Combined stdout+stderr and exit code | **executable allowlist**, no shell metacharacters, timeout + kill |
| `http_get(url)` | Response body | http(s) only, 32 KB cap |
| `search_docs(query)` | Semantically related passages | requires pgvector; registered only when available |
| `spawn_agent(task)` | Delegates to a child run, returns one summary | depth cap 2, bounded budget |

`run_shell` never goes through a shell: the command is `shlex.split` and run with
`asyncio.create_subprocess_exec`, so pipes, redirects, and `;`/`&&` are inert, and the
program must be on the allowlist.

---

## Durability

A crashed run resumes from its newest checkpoint:

```bash
agent run "..."         # prints run_id, then dies / is killed
agent replay <run_id>   # see how far it got
agent resume <run_id>   # continue from there
```

The golden test in `tests/test_checkpointer.py` proves a run interrupted mid-flight and
then resumed lands on the same messages, step count, and status as an uninterrupted run.

---

## Local model and prompt caching (M9)

The harness talks to any OpenAI-compatible endpoint the `anthropic` SDK can reach by
changing one config value:

```bash
ANTHROPIC_BASE_URL=http://localhost:8080  agent run "..."
```

Nothing else changes: `AnthropicClient` passes `base_url` straight through, and the rest
of the harness only sees the `LLMClient` protocol. This was verified end to end against
a local Anthropic-compatible stub server: the model requested `read_file`, the tool ran,
the result went back as a `tool_result` block, and the model answered from the real file
contents — with the run checkpointed, traced, and resumable throughout.

Prompt caching is opt-in with `CACHE_PROMPT=true`. It places one cache breakpoint on the
last tool definition, which is the longest stable prefix shared by every request in a
run, so the provider can reuse it across iterations.

---

## Testing

```bash
pytest                  # whole suite
pytest -m "not slow"    # (no slow marks currently)
ruff check .            # lint
mypy src                # types, strict
```

Tests are deterministic and hit no network: the model is always a scripted
`MockLLMClient` or `ScriptedFixtureLLM`, and files come from `tmp_path` fixtures.
DB-backed tests skip automatically when Postgres is unreachable — start one with:

```bash
docker run --name agent-pg -e POSTGRES_PASSWORD=dev -p 5432:5432 -d postgres:15
```

---

## Measured instrumentation

Collected on this machine (Postgres 14 local, Python 3.12, `scripts/instrument.py`):

| Metric | Result |
|---|---|
| Checkpoint write latency | **p50 1.22 ms / p95 2.85 ms** (n=28) |
| Resume vs. restart, 12-step run | restart 47 ms vs. resume-from-step-10 **20 ms** (2.3× less work) |
| Multi-tool step, 5 × 200 ms tools | sequential 1006 ms vs. concurrent **202 ms** (5.0×) |
| Cost / latency per completed eval task | **$0.001725** avg, 84 ms avg |
| Eval success rate | **100%** baseline; **83%** with a regressed agent |
| Hanging-tool workload ×3, before/after M3 | before: 0 completed / 3 hung → after: **3 completed / 0 crashed** |

Regenerate with `python scripts/instrument.py`.
