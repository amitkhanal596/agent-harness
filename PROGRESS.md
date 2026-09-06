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
