"""Headless eval harness: run labeled tasks and report pass rate, cost, and latency."""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from .agent import Agent, BudgetExceeded, MaxStepsExceeded
from .checkpointer import Checkpointer
from .config import Settings, get_settings
from .context import ContextManager
from .db import get_pool, run_migrations
from .executor import Executor
from .llm import LLMClient
from .models import Completion, Message, ToolCall, ToolSpec
from .otel import configure_tracing
from .tools import build_registry
from .tracer import Tracer


class GradeResult:
    """The outcome of grading one answer, with per-criterion detail.

    A boolean alone makes a failing eval opaque; knowing *which* criterion failed is
    what makes the set actionable when the pass rate drops.
    """

    def __init__(self, passed: bool, failures: list[str] | None = None,
                 score: float = 0.0) -> None:
        self.passed = passed
        self.failures = failures or []
        self.score = score

    def __bool__(self) -> bool:
        return self.passed


class EvalTask:
    """One labeled task and the criteria its answer must satisfy.

    Supported criteria, all optional and combined with AND:

    * `expect_contains` / `expect_absent` — case-insensitive substrings.
    * `expect_regex` — patterns the answer must match, for shapes substrings cannot
      express (a number with any surrounding words, a specific format).
    * `expect_number` — a numeric value with `tolerance`, so "8080" and "port 8,080"
      both pass and "8081" does not; substring matching cannot do this.
    * `rubric` — a natural-language criterion checked by an LLM judge, for answers whose
      correctness is not lexical. Skipped when no judge is configured, so the set stays
      runnable offline.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.id: str = data["id"]
        self.task: str = data["task"]
        self.expect_contains: list[str] = list(data.get("expect_contains", []))
        self.expect_absent: list[str] = list(data.get("expect_absent", []))
        self.expect_regex: list[str] = list(data.get("expect_regex", []))
        self.expect_number: float | None = data.get("expect_number")
        self.tolerance: float = float(data.get("tolerance", 0.0))
        self.rubric: str | None = data.get("rubric")
        self.tags: list[str] = list(data.get("tags", []))

    def grade(self, answer: str) -> GradeResult:
        """Check every configured criterion, reporting which ones failed."""
        failures: list[str] = []
        lowered = answer.lower()

        for needle in self.expect_contains:
            if needle.lower() not in lowered:
                failures.append(f"missing {needle!r}")
        for needle in self.expect_absent:
            if needle.lower() in lowered:
                failures.append(f"unexpectedly present: {needle!r}")
        for pattern in self.expect_regex:
            if not re.search(pattern, answer, re.IGNORECASE):
                failures.append(f"no match for /{pattern}/")
        if self.expect_number is not None:
            numbers = [
                float(n.replace(",", ""))
                for n in re.findall(r"-?\d[\d,]*\.?\d*", answer)
            ]
            if not any(abs(n - self.expect_number) <= self.tolerance for n in numbers):
                failures.append(
                    f"no number within {self.tolerance} of {self.expect_number}"
                )

        checks = (
            len(self.expect_contains) + len(self.expect_absent)
            + len(self.expect_regex) + (1 if self.expect_number is not None else 0)
        )
        score = 1.0 if not checks else (checks - len(failures)) / checks
        return GradeResult(passed=not failures, failures=failures, score=score)

    async def grade_rubric(self, answer: str, judge: LLMClient) -> GradeResult:
        """Judge the answer against `self.rubric` with an LLM.

        The judge is asked for a bare verdict token, so parsing cannot be fooled by an
        explanation that happens to contain the opposite word.
        """
        if not self.rubric:
            return GradeResult(passed=True, score=1.0)
        prompt = (
            "You are grading an AI agent's answer against a criterion.\n\n"
            f"TASK: {self.task}\n\nCRITERION: {self.rubric}\n\n"
            f"ANSWER: {answer}\n\n"
            "Reply with exactly one word: PASS if the answer satisfies the criterion, "
            "FAIL otherwise."
        )
        verdict = await judge.complete([Message(role="user", content=prompt)], [])
        token = verdict.text.strip().upper()
        passed = token.startswith("PASS")
        return GradeResult(
            passed=passed,
            failures=[] if passed else [f"rubric not met: {self.rubric}"],
            score=1.0 if passed else 0.0,
        )


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
    judge: LLMClient | None = None,
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
        configure_tracing(exporter=cfg.otel_exporter, endpoint=cfg.otel_endpoint or None)
        checkpointer = Checkpointer(pool)
        tracer = Tracer(pool, model=cfg.model)
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
                context=ContextManager(
                    max_tokens=cfg.max_context_tokens,
                    keep_recent_messages=cfg.keep_recent_messages,
                ),
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

            grade = task.grade(answer) if error is None else GradeResult(False, ["run failed"])
            if grade.passed and task.rubric and judge is not None:
                grade = await task.grade_rubric(answer, judge)

            results.append(
                {
                    "id": task.id,
                    "run_id": str(state.run_id),
                    "passed": error is None and grade.passed,
                    "score": grade.score,
                    "failures": grade.failures,
                    "tags": task.tags,
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
            cache_prompt=cfg.cache_prompt,
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

    lines = [
        f"{'task':<28} {'pass':<6} {'score':>6} {'steps':>5} {'cost_usd':>10} "
        f"{'latency_ms':>11}  detail"
    ]
    for r in results:
        detail = r["error"] or "; ".join(r.get("failures", []))
        lines.append(
            f"{r['id']:<28} {'PASS' if r['passed'] else 'FAIL':<6} "
            f"{r.get('score', 0.0):>6.2f} {r['steps']:>5} "
            f"{r['cost_usd']:>10.6f} {r['latency_ms']:>11.1f}  {detail}"
        )
    lines.append("")
    lines.append(f"pass rate:   {pass_rate:.0%} ({passed}/{total})")
    if results:
        avg_score = statistics.fmean([r.get("score", 0.0) for r in results])
        lines.append(f"avg score:   {avg_score:.2f}  (partial credit per criterion)")
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
        "avg_score": statistics.fmean([r.get("score", 0.0) for r in results])
        if results
        else 0.0,
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
                input={"command": f"pytest {self.fixture / 'tests'} -q"},
            )
        return ToolCall(
            id="c1", name="read_file", input={"path": str(self.fixture / "requirements.txt")}
        )

    def _answer(self, task: str) -> str:
        if "ci_log.txt" in task:
            if self.sabotage:
                return "The failing test is in the calculator; I could not identify the fix."
            return (
                "The buggy function is apply_discount; it should divide percent by "
                "100, not 10."
            )
        if "config.ini" in task and "workers" in task:
            return "4 workers are configured."
        if "config.ini" in task:
            return "8080"
        if "TODO" in task:
            return "The TODO is in notes.py: handle unicode punctuation."
        if "how many Python files" in task:
            return "2"
        if "run_shell" in task:
            return "1 failed and 3 passed."
        return "httpx"
