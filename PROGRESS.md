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
