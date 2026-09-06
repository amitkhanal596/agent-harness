"""M8: the eval harness — grading, aggregate metrics, and regression sensitivity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import asyncpg

from agent_harness.config import Settings
from agent_harness.evals_runner import (
    EvalTask,
    ScriptedFixtureLLM,
    load_tasks,
    run_eval_set,
)
from agent_harness.models import Completion, Message, ToolSpec

TASKS_FILE = Path(__file__).resolve().parents[1] / "evals" / "tasks.jsonl"
FIXTURE = Path(__file__).resolve().parents[1] / "evals" / "fixture_repo"


def _settings() -> Settings:
    import os

    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=os.environ.get(
            "TEST_DATABASE_URL", "postgres://postgres:dev@localhost:5432/agent"
        ),
        max_steps=5,
        tool_timeout_seconds=30.0,
        tool_max_retries=0,
        run_budget_usd=1.0,
    )


def test_task_set_has_at_least_five_labeled_tasks() -> None:
    tasks = load_tasks(TASKS_FILE)
    assert len(tasks) >= 5
    assert all(t.id and t.task and t.expect_contains for t in tasks)


def test_grading_is_substring_and_case_insensitive() -> None:
    task = EvalTask({"id": "x", "task": "t", "expect_contains": ["Apply_Discount", "100"]})
    assert task.grade("the bug is in apply_discount, divide by 100")
    assert not task.grade("apply_discount looks fine")


def test_grading_honors_expect_absent() -> None:
    task = EvalTask(
        {"id": "x", "task": "t", "expect_contains": ["ok"], "expect_absent": ["traceback"]}
    )
    assert task.grade("ok")
    assert not task.grade("ok, but here is a Traceback")


def test_load_tasks_respects_limit() -> None:
    assert len(load_tasks(TASKS_FILE, limit=2)) == 2


async def test_eval_reports_the_three_aggregate_metrics(db_pool: asyncpg.Pool) -> None:
    report: dict[str, Any] = await run_eval_set(
        TASKS_FILE, llm=ScriptedFixtureLLM(), settings=_settings()
    )
    assert report["total"] >= 5
    assert report["pass_rate"] == 1.0
    assert "pass rate:" in report["text"]
    assert "avg cost:" in report["text"]
    assert "avg latency:" in report["text"]
    assert report["avg_cost_usd"] > 0
    assert report["avg_latency_ms"] > 0
    assert report["checkpoint_p50_ms"] >= 0


async def test_regression_lowers_the_pass_rate(db_pool: asyncpg.Pool) -> None:
    """A deliberately regressed agent scores strictly worse on the same task set."""
    good = await run_eval_set(TASKS_FILE, llm=ScriptedFixtureLLM(), settings=_settings())
    regressed = await run_eval_set(
        TASKS_FILE, llm=ScriptedFixtureLLM(sabotage=True), settings=_settings()
    )
    assert regressed["pass_rate"] < good["pass_rate"]
    assert regressed["passed"] == good["passed"] - 1


async def test_ci_triage_task_finds_the_bug(db_pool: asyncpg.Pool) -> None:
    """The headline M8 task: triage a real failing CI suite in the fixture repo."""
    report = await run_eval_set(
        TASKS_FILE, limit=1, llm=ScriptedFixtureLLM(), settings=_settings()
    )
    result = report["results"][0]
    assert result["id"] == "triage_failing_test"
    assert result["passed"]
    assert "apply_discount" in result["answer"]


async def test_failed_task_does_not_abort_the_set(
    db_pool: asyncpg.Pool, tmp_path: Path
) -> None:
    class BrokenLLM:
        async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
            raise RuntimeError("model exploded")

    tasks_file = tmp_path / "t.jsonl"
    tasks_file.write_text(
        '{"id": "a", "task": "x", "expect_contains": ["x"]}\n'
        '{"id": "b", "task": "y", "expect_contains": ["y"]}\n'
    )
    report = await run_eval_set(tasks_file, llm=BrokenLLM(), settings=_settings())
    assert report["total"] == 2
    assert report["pass_rate"] == 0.0
    assert all("model exploded" in r["error"] for r in report["results"])
