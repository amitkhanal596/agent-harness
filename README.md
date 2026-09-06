# Agent Harness — Implementation Spec

Build a durable, observable agent-execution engine in **Python 3.11+ and PostgreSQL**. This document is the authoritative build spec. Implement it exactly as written, in the milestone order in §8, running the acceptance checks after each milestone before moving on.

This is infrastructure, not a chatbot: a loop that runs an LLM with tools, checkpoints every step to Postgres (resume/replay), executes tools concurrently, and traces cost/latency/failures.

---

## 1. Hard constraints

**MUST**
- Use Python 3.11+ (`asyncio.TaskGroup` and `except*` are required).
- Be async end-to-end. No blocking I/O inside coroutines — use `asyncpg`, `httpx.AsyncClient`, and `asyncio.create_subprocess_exec`. Never call `requests`, `time.sleep`, blocking `subprocess`, or sync file I/O for large files inside a coroutine (`time.sleep` → `asyncio.sleep`).
- Use `pydantic` v2 for every data model. Serialize state with `model_dump_json()` / `model_validate_json()`.
- Validate every tool input against the tool's schema **before** executing it.
- Checkpoint the full state to Postgres **after every loop iteration**.
- Enforce a max-iteration cap and a per-run cost budget.
- Treat a tool failure as data: feed the error back to the model as the tool result. Never let a tool exception crash the loop.
- Keep the layout in §3 and the interfaces in §5 exactly. The signatures in §5 are contracts.
- Add type hints to every function and pass `mypy src`.

**MUST NOT**
- Use any agent/LLM framework (no LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, Haystack, etc.). Implement the loop from scratch. Only the raw provider SDK (`anthropic`) is allowed for model calls.
- Add abstractions, layers, config options, or dependencies not specified here. Prefer the smallest implementation that satisfies the spec.
- Store secrets in code. All secrets come from environment variables.
- Skip ahead: do not start a milestone until the previous milestone's acceptance checks pass.

---

## 2. Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Database | PostgreSQL 15+ (Docker for local dev) |
| DB driver | `asyncpg` |
| Model API | Anthropic via the `anthropic` SDK (behind a `Protocol`, so swappable) |
| HTTP (tools) | `httpx` (async) |
| Validation / schemas | `pydantic` v2 |
| Concurrency | stdlib `asyncio` (`TaskGroup`) |
| Config | `pydantic-settings` (reads env / `.env`) |
| CLI | `typer` |
| Tests | `pytest` + `pytest-asyncio` |
| Lint / format / types | `ruff` + `mypy` |

Local model (optional, Milestone 9): the harness must work unchanged against an OpenAI-compatible `llama.cpp` server by pointing the client `base_url` at it — keep that swap to one config value.

---

## 3. Project layout

Create exactly this structure.

```
agent-harness/
├── pyproject.toml
├── README.md                      # this file
├── .env.example
├── .gitignore                     # .venv, .env, __pycache__, *.egg-info
├── migrations/
│   └── 001_init.sql
├── src/agent_harness/
│   ├── __init__.py
│   ├── config.py                  # Settings (pydantic-settings)
│   ├── models.py                  # State, Message, ToolCall, Completion, ToolSpec, ToolResult
│   ├── db.py                      # asyncpg pool creation + migration runner
│   ├── llm.py                     # LLMClient Protocol + AnthropicClient + cost calc
│   ├── tools/
│   │   ├── __init__.py            # builds and returns a populated Registry
│   │   ├── base.py                # Tool ABC + Registry
│   │   ├── read_file.py
│   │   ├── list_dir.py
│   │   ├── grep.py
│   │   ├── run_shell.py
│   │   └── http_get.py
│   ├── executor.py                # Executor: validate/timeout/retry + run_all (concurrent)
│   ├── checkpointer.py            # Checkpointer: save / latest / list_steps
│   ├── tracer.py                  # Tracer: llm() / tool() event writes + read helpers
│   ├── guardrails.py              # RateLimiter, CircuitBreaker, budget check
│   ├── agent.py                   # Agent.run() loop + resume()
│   └── cli.py                     # typer app: run / resume / replay / trace / eval
├── tests/
│   ├── conftest.py                # fixtures: db pool, MockLLMClient, temp files
│   ├── test_registry.py
│   ├── test_executor.py
│   ├── test_loop.py
│   ├── test_checkpointer.py
│   └── test_concurrency.py
└── evals/
    ├── tasks.jsonl                # labeled eval tasks
    └── run_eval.py                # or expose via `agent eval`
```

`pyproject.toml` must define the package, pin dependencies, expose the CLI as a script (`agent = "agent_harness.cli:app"`), and configure `ruff`/`mypy`/`pytest` (`asyncio_mode = "auto"`).

---

## 4. Configuration & environment

`config.py` exposes a `Settings` object (pydantic-settings). `.env.example` lists every variable:

```
ANTHROPIC_API_KEY=            # required
DATABASE_URL=postgres://postgres:dev@localhost:5432/agent
MODEL=claude-sonnet-4-5       # default model id; confirm the current id before use
MAX_STEPS=25                  # iteration cap
TOOL_TIMEOUT_SECONDS=30
TOOL_MAX_RETRIES=2
RUN_BUDGET_USD=1.00           # abort a run above this
ANTHROPIC_BASE_URL=           # optional; set to a llama.cpp endpoint for local runs
```

Model pricing lives as documented constants in `llm.py` (`PRICE_PER_1M_INPUT`, `PRICE_PER_1M_OUTPUT`) with a comment to verify current pricing. Cost = tokens ÷ 1e6 × price.

---

## 5. Interfaces (contracts — implement exactly)

### 5.1 `models.py`
```python
from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel

class ToolCall(BaseModel):
    id: str
    name: str
    input: dict

class Message(BaseModel):
    role: str                          # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = []    # set when role == "assistant" and it requested tools
    tool_call_id: str | None = None    # set when role == "tool": which call this answers

class State(BaseModel):
    run_id: UUID
    task: str
    messages: list[Message] = []
    step_count: int = 0
    status: str = "running"            # running | completed | failed | paused

class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict                 # JSON Schema

class Completion(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = []
    stop_reason: str                   # "tool_use" | "end_turn" | other
    prompt_tokens: int = 0
    output_tokens: int = 0

class ToolResult(BaseModel):
    call_id: str
    output: str = ""
    error: str | None = None
```

### 5.2 `llm.py`
```python
from typing import Protocol

class LLMClient(Protocol):
    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion: ...
```

`AnthropicClient` implements this. **The message mapping is the highest-risk part — get it right:**
- Send `tools` as `[{"name", "description", "input_schema"}]`. `max_tokens` is required on the request.
- Convert internal `messages` to Anthropic format:
  - `role == "user"` → `{"role": "user", "content": [{"type": "text", "text": ...}]}`.
  - `role == "assistant"` with `tool_calls` → one assistant message whose `content` is an optional `{"type":"text"}` block followed by one `{"type":"tool_use","id","name","input"}` block per tool call.
  - `role == "tool"` → Anthropic has **no** "tool" role. Emit a **user** message containing `{"type":"tool_result","tool_use_id":<tool_call_id>,"content":<output>}`. **Group consecutive tool results into a single user message** (all tool_result blocks for one assistant turn go together), preserving order.
- Parse the response: iterate `response.content`; accumulate `text` blocks into `Completion.text`; map each `tool_use` block to a `ToolCall(id, name, input)`. Set `stop_reason` from `response.stop_reason`, and token counts from `response.usage.input_tokens` / `.output_tokens`.

Reference shape:
```python
from anthropic import AsyncAnthropic

class AnthropicClient:
    def __init__(self, model: str, api_key: str, max_tokens: int = 4096, base_url: str | None = None):
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url or None)
        self._model, self._max_tokens = model, max_tokens

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            tools=[t.model_dump() for t in tools],
            messages=to_anthropic_messages(messages),
        )
        text, calls = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        return Completion(text=text, tool_calls=calls, stop_reason=resp.stop_reason,
                          prompt_tokens=resp.usage.input_tokens,
                          output_tokens=resp.usage.output_tokens)
```

### 5.3 `tools/base.py`
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class Tool(ABC):
    name: str
    description: str
    InputModel: type[BaseModel]                      # each tool sets its own

    def schema(self) -> dict:
        return self.InputModel.model_json_schema()

    @abstractmethod
    async def execute(self, args: BaseModel) -> str: ...

class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)
    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(name=t.name, description=t.description, input_schema=t.schema())
                for t in self._tools.values()]
```

Each tool file defines an `InputModel(BaseModel)` (fields become the JSON schema) and an `execute` that receives the validated model. Tool behavior:
- `read_file(path)` → file contents (cap at N KB; note truncation).
- `list_dir(path)` → newline list of entries.
- `grep(pattern, path)` → matching lines with line numbers (use `asyncio.create_subprocess_exec` with `rg`/`grep`, or pure-Python).
- `run_shell(command)` → stdout+stderr, with a timeout; **restrict to an allowlist of commands** and reject anything else.
- `http_get(url)` → response body via `httpx.AsyncClient` (cap size).

`tools/__init__.py` exposes `build_registry() -> Registry` that instantiates and registers all tools.

### 5.4 `executor.py`
```python
class Executor:
    def __init__(self, registry: Registry, timeout: float, max_retries: int): ...

    async def run(self, call: ToolCall) -> ToolResult:
        # 1. lookup; unknown tool -> ToolResult(error=...)
        # 2. validate call.input via tool.InputModel.model_validate; ValidationError -> ToolResult(error=...)
        # 3. asyncio.wait_for(tool.execute(args), timeout); TimeoutError -> retry
        # 4. retry transient errors with exponential backoff up to max_retries;
        #    non-transient -> return ToolResult(error=...)
        ...

    async def run_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        # concurrent via asyncio.TaskGroup; preserve input order in the result list;
        # a fatal tool error cancels siblings (see Milestone 6)
        ...
```

### 5.5 `checkpointer.py`
```python
class Checkpointer:
    def __init__(self, pool): ...                    # asyncpg pool
    async def create_run(self, task: str) -> State:  # inserts agent_runs row, returns fresh State
        ...
    async def save(self, state: State) -> None:
        # upsert run_checkpoints (run_id, step_number, state_json::jsonb) ON CONFLICT DO UPDATE;
        # update agent_runs summary (status, step_count, totals, updated_at)
        ...
    async def latest(self, run_id) -> State: ...     # newest checkpoint -> State
    async def list_steps(self, run_id) -> list[State]: ...  # all checkpoints ascending (for replay)
```

### 5.6 `tracer.py`
```python
class Tracer:
    def __init__(self, pool): ...
    async def llm(self, state: State, comp: Completion, latency_ms: int, cost_usd: float) -> None: ...
    async def tool(self, state: State, result: ToolResult, tool_name: str, latency_ms: int) -> None: ...
    async def timeline(self, run_id) -> list[dict]: ...   # ordered events for `agent trace`
```
Each call inserts a `run_events` row. Also accumulate `total_tokens` / `total_cost_usd` onto the run.

### 5.7 `agent.py`
```python
class Agent:
    def __init__(self, llm, registry, executor, checkpointer, tracer,
                 max_steps: int, budget_usd: float): ...

    async def run(self, state: State) -> str: ...     # the loop, per §6
    async def resume(self, run_id) -> str:            # load latest checkpoint, continue
        return await self.run(await self.checkpointer.latest(run_id))
```

---

## 6. The loop (authoritative logic)

```
loop while state.step_count < max_steps:
    t0 = now
    comp = await llm.complete(state.messages, registry.specs())
    await tracer.llm(state, comp, latency_ms=now-t0, cost_usd=cost(comp))
    accumulate run cost; if run cost > budget: status="failed"; save; raise BudgetExceeded

    if comp.tool_calls is empty:                       # final answer
        append assistant message(text)
        status = "completed"; save(state); return comp.text

    append assistant message(text, tool_calls=comp.tool_calls)
    results = await executor.run_all(comp.tool_calls)  # concurrent
    for r, call in zip(results, comp.tool_calls):
        content = r.output if r.error is None else f"ERROR: {r.error}"
        append tool message(tool_call_id=r.call_id, content=content)
        await tracer.tool(state, r, tool_name=call.name, latency_ms=...)
    state.step_count += 1
    save(state)                                        # checkpoint every iteration

# cap reached:
status = "failed"; save(state); raise MaxStepsExceeded
```

Rules: never raise out of a tool into the loop (Executor converts errors to `ToolResult.error`). The only exceptions the loop raises are `BudgetExceeded` and `MaxStepsExceeded` (define both). Persist `status` on every exit path.

---

## 7. Database schema — `migrations/001_init.sql`

```sql
CREATE TABLE agent_runs (
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

CREATE TABLE run_checkpoints (
    id          BIGSERIAL PRIMARY KEY,
    run_id      UUID NOT NULL REFERENCES agent_runs(id),
    step_number INT  NOT NULL,
    state_json  JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, step_number)
);

CREATE TABLE run_events (
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

CREATE INDEX idx_checkpoints_run ON run_checkpoints(run_id, step_number);
CREATE INDEX idx_events_run      ON run_events(run_id, step_number);
```

`db.py` provides `async def get_pool()` and `async def run_migrations(pool)` (apply every `.sql` in `migrations/` in filename order; safe to re-run — wrap in `IF NOT EXISTS` or track applied files).

---

## 8. Milestones & acceptance criteria

Implement in order. After each milestone, all of `pytest`, `ruff check`, and `mypy src` must pass, plus the milestone's specific checks. Write the tests for a milestone as part of that milestone.

**M0 — Scaffold.** Create the layout, `pyproject.toml`, `.env.example`, `config.py`, `db.py`, `migrations/001_init.sql`.
- Accept: `agent --help` runs; migrations apply against a local Postgres; `pytest` collects 0 failures.

**M1 — Models + LLM client.** `models.py`, `llm.py` with `AnthropicClient` and the message mapping in §5.2.
- Accept: a unit test with a stubbed SDK response maps a `tool_use` block to `Completion.tool_calls` and an `end_turn` text block to `Completion.text`; token counts populated. A test verifies a `role=="tool"` message serializes to a user `tool_result` block.

**M2 — Registry + tools + minimal sequential loop.** `tools/*`, `Registry`, and a first loop that runs tools sequentially (no DB yet is acceptable here; wire DB in M4).
- Accept: with a `MockLLMClient` scripted to call `read_file` then answer, the agent returns the file's content. Unknown-tool and invalid-input calls produce `ToolResult.error`, not exceptions.

**M3 — Executor reliability.** Timeouts (`asyncio.wait_for`), retry-with-backoff, transient-vs-fatal classification, max-steps cap, `MaxStepsExceeded`.
- Accept: a tool that sleeps past the timeout yields a timeout `ToolResult.error` and the loop continues; a tool that fails twice then succeeds is retried and succeeds; a `MockLLMClient` that never stops trips `MaxStepsExceeded` at the cap.

**M4 — Durable execution.** `checkpointer.py`; loop checkpoints every iteration; `Agent.resume`; `agent replay <run_id>`.
- Accept: a run interrupted after step k, then resumed, reaches the same final state as an uninterrupted run (golden test). `agent replay` prints each checkpoint's step and message count in order. Re-saving the same step number does not error (upsert).

**M5 — Observability.** `tracer.py`; `agent trace <run_id>` prints per-event latency/tokens/cost and run totals; run cost/token totals persisted.
- Accept: after a 2-step run, `run_events` has one `llm_call` per step plus one `tool_call` per tool; `agent trace` totals equal the sum of event rows.

**M6 — Concurrency.** `Executor.run_all` via `asyncio.TaskGroup`; results ordered to match input; fatal error cancels siblings (`FatalToolError` + `except*`), partial results retained.
- Accept: three tools each sleeping ~T run in ≈T not ≈3T (timing assertion); when one raises fatal, the others are cancelled and the returned list still aligns to inputs.

**M7 — Guardrails.** `guardrails.py`: token-bucket `RateLimiter` and `CircuitBreaker` wrapping the LLM client; per-run budget check in the loop (`BudgetExceeded`).
- Accept: N consecutive simulated API failures trip the breaker (subsequent calls fail fast without hitting the client); a run configured with a tiny budget aborts with `BudgetExceeded` and `status="failed"` persisted.

**M8 — Real task + evals.** A CI-failure-triage agent config (tools: `read_file`, `grep`, `run_shell`, `http_get`) run over a fixture repo with a known failing test. `evals/tasks.jsonl` holds ≥5 labeled tasks; `agent eval` reports pass rate, avg cost, avg latency.
- Accept: `agent eval` runs the set headlessly and prints the three aggregate metrics; a deliberately regressed change lowers the pass rate.

**M9 — Optional stretch.** Point `ANTHROPIC_BASE_URL` at a local `llama.cpp` server and run unchanged; add prompt caching; add a sub-agent tool that spawns a child run; add a web timeline for traces.

---

## 9. Required instrumentation

Capture and expose (via `agent trace` / `agent eval`), because they are project requirements, not extras:
- Checkpoint write latency (p50/p95).
- Resume-vs-restart wall-clock for a long run.
- Multi-tool step latency: sequential vs. concurrent.
- Cost and token count per completed task.
- Task success rate on the eval set.
- Completed-vs-crashed run counts before/after M3.

---

## 10. Coding standards

- Full type hints; `from __future__ import annotations` at the top of modules using forward refs. `mypy src` clean.
- `ruff` for lint + format; no unused imports, no bare `except:`.
- Async everywhere; no blocking calls in coroutines. CPU-bound work (if any) goes through `loop.run_in_executor`.
- Small, single-responsibility modules matching §3. No god-objects.
- Brief docstrings on public classes/functions stating purpose and failure behavior.
- Deterministic tests: no live network. Use `MockLLMClient` and local fixtures. Mark DB tests so they can be skipped without Postgres, but provide a `docker run` one-liner in the README section you generate.
- Commit per milestone with a message naming the milestone and its acceptance status.

---

## 11. Definition of done

- All milestones M0–M8 complete; M9 optional.
- `pytest`, `ruff check`, and `mypy src` all pass.
- `agent run "<task>"` completes a real task end-to-end and prints the final answer + run id.
- `agent resume`, `agent replay`, `agent trace`, and `agent eval` all work as specified.
- No agent framework used anywhere; the loop, executor, checkpointer, and concurrency are hand-written.

---

## 12. Local setup (include an executable version of this in generated docs)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker run --name agent-pg -e POSTGRES_PASSWORD=dev -p 5432:5432 -d postgres:15
cp .env.example .env   # fill in ANTHROPIC_API_KEY
# apply migrations (via db.run_migrations or the CLI), then:
agent run "read ./config.txt and tell me the configured port"
```