"""Headless eval harness: run labeled tasks and report pass rate, cost, and latency."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .agent import Agent, BudgetExceeded, MaxStepsExceeded
from .checkpointer import Checkpointer
from .config import Settings, get_settings
from .db import get_pool, run_migrations
from .executor import Executor
from .llm import LLMClient
from .models import Completion, Message, ToolCall, ToolSpec
from .tools import build_registry
from .tracer import Tracer


class EvalTask:
    """One labeled task: a prompt plus the substrings its answer must contain."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.id: str = data["id"]
        self.task: str = data["task"]
        self.expect_contains: list[str] = list(data.get("expect_contains", []))
        self.expect_absent: list[str] = list(data.get("expect_absent", []))

    def grade(self, answer: str) -> bool:
        """Case-insensitive substring grading against the labels."""
        lowered = answer.lower()
        if not all(needle.lower() in lowered for needle in self.expect_contains):
            return False
        return all(needle.lower() not in lowered for needle in self.expect_absent)


def load_tasks(path: str | Path, limit: int | None = None) -> list[EvalTask]:
    """Read a JSONL task file, skipping blank lines."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    tasks = [EvalTask(json.loads(ln)) for ln in lines]
    return tasks[:limit] if limit else tasks


async def run_eval_set(
    path: str | Path,
    limit: int | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run every task headlessly and return per-task results plus aggregate metrics.

    A task that raises `BudgetExceeded` / `MaxStepsExceeded` counts as a failure rather
    than aborting the set.
    """
    cfg = settings or get_settings()
    tasks = load_tasks(path, limit)
    pool = await get_pool(cfg.database_url)
    results: list[dict[str, Any]] = []
    checkpoint_latencies: list[float] = []
    try:
        await run_migrations(pool)
        registry = build_registry(
            shell_timeout=cfg.tool_timeout_seconds, http_timeout=cfg.tool_timeout_seconds
        )
        checkpointer = Checkpointer(pool)
        tracer = Tracer(pool)
        client = llm if llm is not None else _default_client(cfg)

        for task in tasks:
            agent = Agent(
                llm=client,
                registry=registry,
                executor=Executor(
                    registry,
                    timeout=cfg.tool_timeout_seconds,
                    max_retries=cfg.tool_max_retries,
                ),
                checkpointer=checkpointer,
                tracer=tracer,
                max_steps=cfg.max_steps,
                budget_usd=cfg.run_budget_usd,
            )
            state = await checkpointer.create_run(task.task)
            t0 = time.perf_counter()
            error: str | None = None
            answer = ""
            try:
                answer = await agent.run(state)
            except (BudgetExceeded, MaxStepsExceeded) as exc:
                error = str(exc)
                await checkpointer.mark_failed(state.run_id, error)
            except Exception as exc:  # noqa: BLE001 - one bad task must not abort the set
                error = f"{type(exc).__name__}: {exc}"
                await checkpointer.mark_failed(state.run_id, error)
            latency_ms = (time.perf_counter() - t0) * 1000
            results.append(
                {
                    "id": task.id,
                    "run_id": str(state.run_id),
                    "passed": error is None and task.grade(answer),
                    "error": error,
                    "answer": answer,
                    "latency_ms": latency_ms,
                    "cost_usd": agent.total_cost_usd,
                    "steps": state.step_count,
                }
            )
        checkpoint_latencies = list(checkpointer.write_latencies_ms)
    finally:
        await pool.close()

    return _summarize(results, checkpoint_latencies)


def _default_client(cfg: Settings) -> LLMClient:
    from .guardrails import CircuitBreaker, GuardedLLMClient, RateLimiter
    from .llm import AnthropicClient

    if not cfg.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in .env)")
    return GuardedLLMClient(
        AnthropicClient(
            model=cfg.model,
            api_key=cfg.anthropic_api_key,
            base_url=cfg.anthropic_base_url or None,
        ),
        RateLimiter(rate=2.0, capacity=5),
        CircuitBreaker(),
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def _summarize(
    results: list[dict[str, Any]], checkpoint_latencies: list[float]
) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    pass_rate = passed / total if total else 0.0
    avg_cost = statistics.fmean([r["cost_usd"] for r in results]) if results else 0.0
    avg_latency = statistics.fmean([r["latency_ms"] for r in results]) if results else 0.0

    lines = [f"{'task':<28} {'pass':<6} {'steps':>5} {'cost_usd':>10} {'latency_ms':>11}  error"]
    for r in results:
        lines.append(
            f"{r['id']:<28} {'PASS' if r['passed'] else 'FAIL':<6} {r['steps']:>5} "
            f"{r['cost_usd']:>10.6f} {r['latency_ms']:>11.1f}  {r['error'] or ''}"
        )
    lines.append("")
    lines.append(f"pass rate:   {pass_rate:.0%} ({passed}/{total})")
    lines.append(f"avg cost:    ${avg_cost:.6f}")
    lines.append(f"avg latency: {avg_latency:.1f} ms")
    if checkpoint_latencies:
        lines.append(
            f"checkpoint write latency: p50={_percentile(checkpoint_latencies, 50):.2f}ms "
            f"p95={_percentile(checkpoint_latencies, 95):.2f}ms"
        )

    return {
        "results": results,
        "total": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "avg_cost_usd": avg_cost,
        "avg_latency_ms": avg_latency,
        "checkpoint_p50_ms": _percentile(checkpoint_latencies, 50),
        "checkpoint_p95_ms": _percentile(checkpoint_latencies, 95),
        "text": "\n".join(lines),
    }


FIXTURE_REPO = Path(__file__).resolve().parents[2] / "evals" / "fixture_repo"


class ScriptedFixtureLLM:
    """An offline stand-in model that solves the bundled eval tasks with real tool calls.

    It exists so `agent eval --mock` and the test suite can exercise the whole harness
    without network access. `sabotage=True` answers the CI-triage task incorrectly,
    which is how the regression check in M8 lowers the pass rate.
    """

    def __init__(self, sabotage: bool = False, fixture: Path | None = None) -> None:
        self.sabotage = sabotage
        self.fixture = fixture or FIXTURE_REPO

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        task = messages[0].content
        if not any(m.role == "tool" for m in messages):
            return Completion(
                stop_reason="tool_use",
                prompt_tokens=100,
                output_tokens=25,
                tool_calls=[self._first_call(task)],
            )
        return Completion(
            stop_reason="end_turn",
            prompt_tokens=200,
            output_tokens=30,
            text=self._answer(task),
        )

    def _first_call(self, task: str) -> ToolCall:
        if "ci_log.txt" in task:
            return ToolCall(
                id="c1", name="read_file", input={"path": str(self.fixture / "ci_log.txt")}
            )
        if "config.ini" in task:
            return ToolCall(
                id="c1", name="read_file", input={"path": str(self.fixture / "config.ini")}
            )
        if "TODO" in task:
            return ToolCall(
                id="c1", name="grep", input={"pattern": "TODO", "path": str(self.fixture)}
            )
        if "how many Python files" in task:
            return ToolCall(id="c1", name="list_dir", input={"path": str(self.fixture / "src")})
        if "run_shell" in task:
            return ToolCall(
                id="c1",
                name="run_shell",
                input={"command": f"python -m pytest {self.fixture / 'tests'} -q"},
            )
        return ToolCall(
            id="c1", name="read_file", input={"path": str(self.fixture / "requirements.txt")}
        )

    def _answer(self, task: str) -> str:
        if "ci_log.txt" in task:
            if self.sabotage:
                return "The failing test is in the calculator; I could not identify the fix."
            return "The buggy function is apply_discount; it should divide percent by 100, not 10."
        if "config.ini" in task:
            return "8080"
        if "TODO" in task:
            return "The TODO is in notes.py: handle unicode punctuation."
        if "how many Python files" in task:
            return "2"
        if "run_shell" in task:
            return "1 test failed and 3 passed."
        return "httpx"
