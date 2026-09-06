# Build Progress — Agent Harness

Running summary of work completed against the spec in `README.md`.
Gate for every milestone: `pytest`, `ruff check`, `mypy src` all pass, plus the
milestone's own acceptance checks.

## Environment notes
- Python 3.12.14 in `.venv` (spec requires 3.11+ for `TaskGroup` / `except*`).
- Docker daemon is not running on this machine; used the local Homebrew
  **PostgreSQL 14** server instead, creating role `postgres/dev` and database
  `agent` so the spec's `DATABASE_URL` works verbatim. PG14 is sufficient —
  the only version-sensitive feature used, `gen_random_uuid()`, is core since PG13.

## Milestones

### M0 — Scaffold ✅
- Created the exact §3 layout, `pyproject.toml` (deps pinned, `agent` script,
  ruff/mypy/pytest config with `asyncio_mode = "auto"`), `.env.example`, `.gitignore`.
- `config.py` (pydantic-settings `Settings`), `db.py` (`get_pool`, `run_migrations`
  tracking applied files in `schema_migrations`), `migrations/001_init.sql` (§7 schema).
- Acceptance: `agent --help` runs; `agent migrate` applies migrations and is
  idempotent on re-run; `pytest` 3 passed, `ruff check` clean, `mypy src` clean.

### M1 — Models + LLM client ✅
- `models.py`: `ToolCall`, `Message`, `State`, `ToolSpec`, `Completion`, `ToolResult`
  exactly per §5.1 (bare `dict` widened to `dict[str, Any]` so `mypy --strict` passes;
  same JSON contract).
- `llm.py`: `LLMClient` Protocol, `AnthropicClient`, `to_anthropic_messages`, and
  documented pricing constants + `cost_usd()`.
- Message mapping handles the tricky part: no "tool" role on the wire — tool results
  become `tool_result` blocks in a **user** message, and consecutive results are grouped
  into one user message in order.
- Acceptance: stubbed SDK response maps a `tool_use` block to `Completion.tool_calls`
  and an `end_turn` text block to `Completion.text` with token counts populated; a
  `role=="tool"` message serializes to a user `tool_result` block. 11 tests pass.

### M2 — Registry + tools + loop ✅
- `tools/base.py` (`Tool` ABC + `Registry`) and all five tools: `read_file` (64 KB cap,
  truncation noted), `list_dir`, `grep` (pure-Python regex over a file or tree),
  `run_shell` (executable allowlist, no shell metacharacters, timeout + kill),
  `http_get` (scheme check, 32 KB cap). `build_registry()` wires them up.
- `executor.py`: lookup → schema validation → `asyncio.wait_for` → retry.
- `agent.py`: the §6 loop plus `BudgetExceeded` / `MaxStepsExceeded`; checkpointer and
  tracer are optional collaborators (null objects) so the loop runs before the DB lands.
- Blocking file I/O is pushed to threads via `asyncio.to_thread`; subprocesses use
  `asyncio.create_subprocess_exec`.
- Acceptance: scripted `read_file` → answer returns the file's contents and the tool
  output is provably fed back to the model; unknown-tool and invalid-input calls produce
  `ToolResult.error` rather than exceptions. 36 tests pass.

### M3 — Executor reliability ✅
- Per-call `asyncio.wait_for` timeout, exponential backoff (0.1 · 2ⁿ), and
  transient-vs-fatal classification (`TimeoutError`/`ConnectionError`/`OSError` retry;
  everything else fails immediately; `FatalToolError` never retries).
- Max-step cap enforced in the loop with `MaxStepsExceeded` and `status="failed"`.
- Acceptance: a tool sleeping past the timeout yields `ERROR: slow timed out` and the
  loop continues to a final answer; a tool failing twice then succeeding is retried
  (3 attempts, `succeeded`); a never-stopping MockLLMClient trips `MaxStepsExceeded`
  exactly at the cap. Backoff delays asserted as [0.1, 0.2, 0.4]. 47 tests pass.

### M4 — Durable execution ✅
- `checkpointer.py`: `create_run` / `save` (upsert on `(run_id, step_number)` + run
  summary refresh) / `latest` / `list_steps` / `mark_failed`, with write latencies
  recorded for the §9 metrics.
- Loop checkpoints after every iteration and on every exit path; `Agent.resume(run_id)`
  reloads the newest checkpoint; `agent replay <run_id>` added.
- Acceptance (against real Postgres): golden test — a run crashed after step 2 and then
  resumed reaches byte-identical messages, step count, and status as an uninterrupted
  run. `agent replay` prints each checkpoint's step and message count in order
  (0/1 msg → 1/3 msgs → 2/6 msgs). Re-saving the same step number upserts, no error.

### M5 — Observability ✅
- `tracer.py`: one `run_events` row per model call and per tool call, run-level
  `total_tokens` / `total_cost_usd` accumulated on `agent_runs`, plus `timeline()` and
  `totals()` read helpers.
- `agent trace <run_id>` prints a per-event table (step, kind, tool, latency, tokens,
  cost, error) and both event-sum and persisted run totals.
- Acceptance: after a 2-step run `run_events` holds exactly one `llm_call` per step plus
  one `tool_call` per tool; `agent trace` event totals equal the persisted run totals
  (verified live: 5 events, 4325 tokens, $0.014475 both ways).

### M6 — Concurrency ✅
- `Executor.run_all` fans out over `asyncio.TaskGroup`, writing into a pre-sized slot
  list so results always align with the input order regardless of finish order.
- `FatalToolError` + `except*`: a fatal tool cancels its siblings; calls that already
  finished keep their results, cancelled ones report `cancelled: sibling tool raised …`.
- Acceptance: three ~300 ms tools finish in ≈300 ms not ≈900 ms (timing assertion);
  a fatal sibling cancels two 2 s tools in <1 s with the result list still aligned to
  the inputs; partial results retained.
- §9 instrumentation: 5 tools × 200 ms → **sequential 1007 ms vs concurrent 202 ms
  (5.0× speedup)**.

### M7 — Guardrails ✅
- `guardrails.py`: token-bucket `RateLimiter` (lock-protected, burst to capacity),
  `CircuitBreaker` (closed → open at threshold → half-open after cooldown),
  `check_budget`, and `GuardedLLMClient` wrapping any `LLMClient`.
- The loop enforces the per-run budget after every model call, persisting
  `status="failed"` before raising `BudgetExceeded`.
- Acceptance: 3 consecutive simulated API failures trip the breaker and the next 4 calls
  fail fast with `CircuitOpen` **without reaching the client** (asserted via a call
  counter that stays at 3); a run with a $0.000001 budget aborts with `BudgetExceeded`
  and `status="failed"` persisted in both the checkpoint and the `agent_runs` row.

### M8 — Real task + evals ✅
- Fixture repo at `evals/fixture_repo/`: a calculator library whose `apply_discount`
  divides by 10 instead of 100, a pytest suite where `test_apply_discount` genuinely
  fails (verified: `1 failed, 3 passed`), the captured `ci_log.txt`, plus `config.ini`,
  `requirements.txt`, and a `TODO` for the other tasks.
- `evals/tasks.jsonl`: **6 labeled tasks** (spec requires ≥5), including the headline
  CI-failure-triage task that reads the CI log, then the source, and names the buggy
  function. Tools exercised: `read_file`, `grep`, `run_shell`, `list_dir`.
- `evals_runner.py` + `evals/run_eval.py` + `agent eval`: substring/absence grading,
  pass rate, avg cost, avg latency, and checkpoint p50/p95. One failing task never
  aborts the set.
- `agent eval --mock` runs the whole set offline against `ScriptedFixtureLLM`, so the
  command is demonstrable without an API key.
- Acceptance: `agent eval` prints all three aggregate metrics (100% pass, $0.001725 avg
  cost, 47.1 ms avg latency); a deliberately regressed agent drops the pass rate
  **100% → 83%** (the triage task flips to FAIL).

### Bug found and fixed along the way
`pydantic-settings` read the `.env` inline comments as values — `ANTHROPIC_API_KEY`
came back as the literal string `# required`, so `agent config` reported a key as set
when none was. Added a `field_validator` that strips a `#` comment only when it starts
the line or follows whitespace, leaving a `#` inside a URL or password intact.

### M9 — Optional stretch (partially done) ✅
- **Local model swap**: verified end to end. Running the real `AnthropicClient` against
  a local Anthropic-compatible stub server via `ANTHROPIC_BASE_URL` alone, `agent run`
  completed the spec's own example task — the model requested `read_file`, the tool ran,
  the result returned as a `tool_result` block, and the answer came from the real file
  ("The configured port is 8080."). Tests lock the pass-through in.
- **Prompt caching**: opt-in `CACHE_PROMPT=true` puts one cache breakpoint on the last
  tool definition (the stable prefix of every request in a run).
- Not done (explicitly optional): sub-agent tool, web timeline UI.

## Definition of done — verified
- M0–M8 complete, each with its acceptance checks passing before moving on.
- `pytest` **93 passed**, `ruff check` clean, `mypy src` clean (strict).
- `agent run "read ./config.txt and tell me the configured port"` completes end to end
  and prints the final answer plus the run id.
- `agent resume` continued a genuinely interrupted run (step 2 → completed);
  `agent replay`, `agent trace`, and `agent eval` all work as specified.
- No agent framework anywhere — loop, executor, checkpointer, and concurrency are all
  hand-written; `anthropic` is the only model dependency, behind an `LLMClient` Protocol.

## Docs produced
- `DOCS.md` — operator's guide: setup, commands, config table, loop diagram, tool
  guardrails, durability, local-model swap, testing, and the measured metrics.
- `scripts/setup.sh` — executable, idempotent version of README §12.
- `scripts/instrument.py` — regenerates the §9 instrumentation table.

## §9 instrumentation (measured, `scripts/instrument.py`)
| Metric | Result |
|---|---|
| Checkpoint write latency | p50 1.22 ms / p95 2.85 ms (n=28) |
| Resume vs restart, 12-step run | restart 47 ms vs resume-from-step-10 20 ms (2.3× less work) |
| Multi-tool step, 5 × 200 ms | sequential 1006 ms vs concurrent 202 ms (5.0×) |
| Cost / latency per completed eval task | $0.001725 avg, ~50 ms avg |
| Eval success rate | 100% baseline, 83% regressed |
| Hanging-tool workload ×3, before/after M3 | 0 completed / 3 hung → 3 completed / 0 crashed |

## Final audit
- §3 layout: every specified file present, with two additions —
  `src/agent_harness/evals_runner.py` (the engine both `agent eval` and
  `evals/run_eval.py` call) and `tests/helpers.py` (the shared `MockLLMClient`, since
  `tests/` is not an importable package).
- §1 MUST NOTs: no LangChain / LangGraph / LlamaIndex / CrewAI / AutoGen / Haystack
  anywhere; no `requests`, `time.sleep`, or blocking `subprocess` in `src/`; no bare
  `except:`; no secrets in code; `.env` git-ignored.
- State serialization goes through `model_dump_json()` / `model_validate_json()`.
- `mypy --strict src` clean; every function in `src/` is typed.
- Tests are deterministic and hit no network. Without Postgres the suite still runs:
  **76 passed, 17 skipped, 0 failed**.

## How to verify this build
```bash
./scripts/setup.sh                  # venv, install, Postgres check, migrations
./.venv/bin/pytest -q               # 93 passed
./.venv/bin/ruff check .            # clean
./.venv/bin/mypy src                # clean (strict)
./.venv/bin/agent eval --mock       # 100% pass rate, offline
./.venv/bin/python scripts/instrument.py   # regenerate the §9 metrics
```
