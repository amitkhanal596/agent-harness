"""Collect the instrumentation README §9 requires.

Run from the project root with the venv python:

    ./.venv/bin/python scripts/instrument.py

Uses scripted models only — no API key and no network needed — but does write to the
configured Postgres database.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import tempfile
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

sys.path.insert(0, "tests")

from agent_harness.agent import Agent  # noqa: E402
from agent_harness.checkpointer import Checkpointer  # noqa: E402
from agent_harness.config import get_settings  # noqa: E402
from agent_harness.db import get_pool, run_migrations  # noqa: E402
from agent_harness.evals_runner import ScriptedFixtureLLM, run_eval_set  # noqa: E402
from agent_harness.executor import Executor  # noqa: E402
from agent_harness.models import Completion, State, ToolCall  # noqa: E402
from agent_harness.tools import build_registry  # noqa: E402
from agent_harness.tools.base import Registry, Tool  # noqa: E402
from agent_harness.tracer import Tracer  # noqa: E402
from helpers import MockLLMClient  # noqa: E402

LONG_RUN_STEPS = 12
FANOUT_TOOLS = 5
FANOUT_SLEEP = 0.2


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


class SleepInput(BaseModel):
    seconds: float = FANOUT_SLEEP


class SleeperTool(Tool):
    name = "sleeper"
    description = "sleeps for `seconds`"
    InputModel = SleepInput

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, SleepInput)
        await asyncio.sleep(args.seconds)
        return "ok"


class HangingTool(Tool):
    name = "hang"
    description = "never returns in time"
    InputModel = SleepInput

    async def execute(self, args: BaseModel) -> str:
        await asyncio.sleep(30)
        return "never"


class CrashingLLM(MockLLMClient):
    """Serves `crash_after` completions then raises, simulating a process death."""

    def __init__(self, script: list[Completion], crash_after: int) -> None:
        super().__init__(script)
        self.crash_after = crash_after
        self.served = 0

    async def complete(self, messages: Any, tools: Any) -> Completion:
        if self.served >= self.crash_after:
            raise RuntimeError("simulated crash")
        self.served += 1
        return await super().complete(messages, tools)


def long_script(path: str) -> list[Completion]:
    script = [
        Completion(
            stop_reason="tool_use",
            prompt_tokens=800,
            output_tokens=40,
            tool_calls=[ToolCall(id=f"t{i}", name="read_file", input={"path": path})],
        )
        for i in range(LONG_RUN_STEPS)
    ]
    script.append(
        Completion(stop_reason="end_turn", text="done", prompt_tokens=900, output_tokens=25)
    )
    return script


def hang_script() -> list[Completion]:
    return [
        Completion(
            stop_reason="tool_use",
            prompt_tokens=10,
            output_tokens=5,
            tool_calls=[ToolCall(id="h", name="hang", input={})],
        ),
        Completion(stop_reason="end_turn", text="recovered", prompt_tokens=10, output_tokens=5),
    ]


async def main() -> None:
    settings = get_settings()
    pool = await get_pool(settings.database_url)
    await run_migrations(pool)

    tmp = pathlib.Path(tempfile.mkdtemp())
    fixture = tmp / "config.txt"
    fixture.write_text("port=8080\n")

    registry = build_registry()
    checkpointer = Checkpointer(pool)
    tracer = Tracer(pool)

    def agent(llm: object, max_steps: int = 40) -> Agent:
        return Agent(
            llm=llm,  # type: ignore[arg-type]
            registry=registry,
            executor=Executor(registry, timeout=5.0, max_retries=1),
            checkpointer=checkpointer,
            tracer=tracer,
            max_steps=max_steps,
            budget_usd=100.0,
        )

    # 1 + 2: checkpoint latency, and resume vs restart for a long run.
    checkpointer.write_latencies_ms.clear()
    baseline = await checkpointer.create_run("long run baseline")
    t0 = time.perf_counter()
    await agent(MockLLMClient(long_script(str(fixture)))).run(baseline)
    restart_s = time.perf_counter() - t0

    interrupted = await checkpointer.create_run("long run interrupted")
    try:
        await agent(CrashingLLM(long_script(str(fixture)), LONG_RUN_STEPS - 2)).run(interrupted)
    except RuntimeError:
        pass
    t0 = time.perf_counter()
    await agent(MockLLMClient(long_script(str(fixture))[LONG_RUN_STEPS - 2 :])).resume(
        interrupted.run_id
    )
    resume_s = time.perf_counter() - t0
    latencies = list(checkpointer.write_latencies_ms)

    # 3: sequential vs concurrent multi-tool step.
    fanout_registry = Registry()
    fanout_registry.register(SleeperTool())
    fanout = Executor(fanout_registry, timeout=5.0, max_retries=0)
    calls = [
        ToolCall(id=f"c{i}", name="sleeper", input={"seconds": FANOUT_SLEEP})
        for i in range(FANOUT_TOOLS)
    ]
    t0 = time.perf_counter()
    for call in calls:
        await fanout.run(call)
    sequential_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    await fanout.run_all(calls)
    concurrent_s = time.perf_counter() - t0

    # 6: completed vs crashed runs on a hanging tool, before and after M3's timeout.
    hang_registry = Registry()
    hang_registry.register(HangingTool())

    async def hang_trial(tool_timeout: float, wall_clock_cap: float | None) -> bool:
        run = Agent(
            llm=MockLLMClient(hang_script()),
            registry=hang_registry,
            executor=Executor(hang_registry, timeout=tool_timeout, max_retries=0),
            max_steps=5,
            budget_usd=100.0,
        ).run(State(run_id=uuid4(), task="hang"))
        try:
            if wall_clock_cap is not None:
                async with asyncio.timeout(wall_clock_cap):
                    await run
            else:
                await run
        except Exception:
            return False
        return True

    after_ok = sum([await hang_trial(0.05, None) for _ in range(3)])
    before_ok = sum([await hang_trial(3600, 0.3) for _ in range(3)])

    await pool.close()

    # 4 + 5: cost, latency, and success rate on the eval set.
    baseline_report = await run_eval_set("evals/tasks.jsonl", llm=ScriptedFixtureLLM())
    regressed_report = await run_eval_set(
        "evals/tasks.jsonl", llm=ScriptedFixtureLLM(sabotage=True)
    )

    print("## §9 instrumentation\n")
    print(
        f"1. Checkpoint write latency: p50={percentile(latencies, 50):.2f}ms "
        f"p95={percentile(latencies, 95):.2f}ms (n={len(latencies)})"
    )
    print(
        f"2. Long run ({LONG_RUN_STEPS} steps): restart={restart_s * 1000:.0f}ms vs "
        f"resume-from-step-{LONG_RUN_STEPS - 2}={resume_s * 1000:.0f}ms "
        f"({restart_s / resume_s:.1f}x less work)"
    )
    print(
        f"3. Multi-tool step ({FANOUT_TOOLS}x{int(FANOUT_SLEEP * 1000)}ms): "
        f"sequential={sequential_s * 1000:.0f}ms vs concurrent={concurrent_s * 1000:.0f}ms "
        f"({sequential_s / concurrent_s:.1f}x)"
    )
    print(
        f"4. Per completed eval task: ${baseline_report['avg_cost_usd']:.6f} avg cost, "
        f"{baseline_report['avg_latency_ms']:.1f}ms avg latency"
    )
    print(
        f"5. Eval success rate: {baseline_report['pass_rate']:.0%} baseline, "
        f"{regressed_report['pass_rate']:.0%} regressed"
    )
    print(
        f"6. Hanging-tool workload x3: before M3 (no timeout) completed={before_ok} "
        f"crashed/hung={3 - before_ok} | after M3 completed={after_ok} crashed={3 - after_ok}"
    )


if __name__ == "__main__":
    asyncio.run(main())
