"""§3.2: richer eval grading — regex, numeric tolerance, rubrics, partial credit."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.evals_runner import EvalTask, ScriptedFixtureLLM, load_tasks, run_eval_set
from agent_harness.models import Completion, Message, ToolSpec

TASKS_FILE = Path(__file__).resolve().parents[1] / "evals" / "tasks.jsonl"


class _Judge:
    """A scripted LLM judge."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.prompts: list[str] = []

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        self.prompts.append(messages[0].content)
        return Completion(stop_reason="end_turn", text=self.verdict)


# --- substring (unchanged behaviour) --------------------------------------------------


def test_substring_criteria_still_work() -> None:
    task = EvalTask({"id": "x", "task": "t", "expect_contains": ["apply_discount"]})
    assert task.grade("the bug is in apply_discount").passed
    assert not task.grade("no mention here").passed


def test_absent_criteria_still_work() -> None:
    task = EvalTask({"id": "x", "task": "t", "expect_absent": ["traceback"]})
    assert task.grade("all good").passed
    assert not task.grade("saw a Traceback").passed


# --- regex ----------------------------------------------------------------------------


def test_regex_matches_shapes_substrings_cannot() -> None:
    task = EvalTask({"id": "x", "task": "t", "expect_regex": [r"1\s+failed", r"3\s+passed"]})
    assert task.grade("Result: 1 failed, 3 passed in 0.03s").passed
    assert task.grade("1    failed and 3     passed").passed
    assert not task.grade("2 failed, 3 passed").passed


def test_regex_is_case_insensitive() -> None:
    task = EvalTask({"id": "x", "task": "t", "expect_regex": ["APPLY_DISCOUNT"]})
    assert task.grade("apply_discount is buggy").passed


# --- numeric tolerance ----------------------------------------------------------------


def test_numeric_criterion_ignores_surrounding_prose() -> None:
    """Substring matching cannot express 'the number 8080, however it is phrased'."""
    task = EvalTask({"id": "x", "task": "t", "expect_number": 8080})
    assert task.grade("8080").passed
    assert task.grade("The service listens on port 8080.").passed
    assert task.grade("port 8,080").passed  # thousands separator
    assert not task.grade("port 8081").passed


def test_numeric_tolerance_is_respected() -> None:
    task = EvalTask({"id": "x", "task": "t", "expect_number": 100, "tolerance": 5})
    assert task.grade("about 103").passed
    assert not task.grade("about 120").passed


def test_numeric_criterion_handles_no_numbers() -> None:
    task = EvalTask({"id": "x", "task": "t", "expect_number": 42})
    result = task.grade("there is no number here")
    assert not result.passed
    assert "no number within" in result.failures[0]


# --- failure detail and partial credit -------------------------------------------------


def test_failures_name_the_criterion_that_failed() -> None:
    """A bare boolean makes a failing eval opaque; the detail is what makes it actionable."""
    task = EvalTask(
        {"id": "x", "task": "t", "expect_contains": ["alpha", "beta"],
         "expect_absent": ["gamma"]}
    )
    result = task.grade("only alpha and gamma")
    assert not result.passed
    assert any("beta" in f for f in result.failures)
    assert any("gamma" in f for f in result.failures)


def test_partial_credit_reflects_how_close_the_answer_was() -> None:
    task = EvalTask({"id": "x", "task": "t", "expect_contains": ["a", "b", "c", "d"]})
    assert task.grade("a b c d").score == 1.0
    assert task.grade("a b c").score == pytest.approx(0.75)
    assert task.grade("nothing").score == 0.0


def test_task_with_no_criteria_scores_full() -> None:
    assert EvalTask({"id": "x", "task": "t"}).grade("anything").score == 1.0


def test_grade_result_is_truthy() -> None:
    task = EvalTask({"id": "x", "task": "t", "expect_contains": ["yes"]})
    assert bool(task.grade("yes")) is True
    assert bool(task.grade("no")) is False


# --- LLM-as-judge ----------------------------------------------------------------------


async def test_rubric_passes_when_the_judge_says_pass() -> None:
    task = EvalTask({"id": "x", "task": "find the bug", "rubric": "identifies the function"})
    judge = _Judge("PASS")
    result = await task.grade_rubric("the bug is in apply_discount", judge)
    assert result.passed
    assert result.score == 1.0
    # The judge saw the task, the criterion, and the answer.
    assert "find the bug" in judge.prompts[0]
    assert "identifies the function" in judge.prompts[0]
    assert "apply_discount" in judge.prompts[0]


async def test_rubric_fails_when_the_judge_says_fail() -> None:
    task = EvalTask({"id": "x", "task": "t", "rubric": "must name the function"})
    result = await task.grade_rubric("I could not determine the cause", _Judge("FAIL"))
    assert not result.passed
    assert "rubric not met" in result.failures[0]


async def test_judge_verdict_parsing_is_not_fooled_by_explanations() -> None:
    """A verdict of 'FAIL - it does not pass the criterion' must not read as PASS."""
    task = EvalTask({"id": "x", "task": "t", "rubric": "r"})
    result = await task.grade_rubric("answer", _Judge("FAIL - it does not pass"))
    assert not result.passed


async def test_no_rubric_is_a_pass() -> None:
    task = EvalTask({"id": "x", "task": "t"})
    assert (await task.grade_rubric("anything", _Judge("FAIL"))).passed


# --- task file and end-to-end ----------------------------------------------------------


def test_task_file_uses_the_richer_criteria() -> None:
    tasks = load_tasks(TASKS_FILE)
    assert len(tasks) >= 5
    assert any(t.expect_regex for t in tasks), "expected at least one regex criterion"
    assert any(t.expect_number is not None for t in tasks), "expected a numeric criterion"
    assert any(t.rubric for t in tasks), "expected at least one rubric"
    assert all(t.tags for t in tasks), "every task should be tagged"


async def test_eval_set_reports_score_alongside_pass_rate(db_pool: object) -> None:
    report = await run_eval_set(TASKS_FILE, llm=ScriptedFixtureLLM())
    assert report["pass_rate"] == 1.0
    assert report["avg_score"] == 1.0
    assert "avg score:" in report["text"]
    assert all("score" in r for r in report["results"])


async def test_regressed_agent_shows_which_criterion_broke(db_pool: object) -> None:
    """The regression must be diagnosable, not just visible as a lower number."""
    report = await run_eval_set(TASKS_FILE, llm=ScriptedFixtureLLM(sabotage=True))
    assert report["pass_rate"] < 1.0
    failed = [r for r in report["results"] if not r["passed"]]
    assert failed
    assert failed[0]["failures"], "a failing task must say which criterion failed"
    assert 0.0 <= failed[0]["score"] < 1.0


async def test_rubric_is_used_when_a_judge_is_supplied(db_pool: object) -> None:
    report = await run_eval_set(
        TASKS_FILE, limit=1, llm=ScriptedFixtureLLM(), judge=_Judge("FAIL")
    )
    # The first task has a rubric; a FAIL verdict must flip it despite lexical criteria.
    assert report["pass_rate"] == 0.0
    assert "rubric not met" in report["results"][0]["failures"][0]


async def test_rubric_is_skipped_without_a_judge(db_pool: object) -> None:
    """The set must stay runnable offline; rubrics are simply not evaluated."""
    report = await run_eval_set(TASKS_FILE, limit=1, llm=ScriptedFixtureLLM())
    assert report["pass_rate"] == 1.0
