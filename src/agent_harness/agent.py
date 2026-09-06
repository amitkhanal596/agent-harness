"""The agent loop: model call, tool fan-out, checkpoint, repeat."""

from __future__ import annotations

import time
from typing import Protocol
from uuid import UUID

from .executor import Executor
from .llm import LLMClient, cost_usd
from .models import Completion, Message, State, ToolResult
from .tools.base import Registry


class BudgetExceeded(Exception):
    """The run's accumulated model spend passed the configured budget."""


class MaxStepsExceeded(Exception):
    """The run hit its iteration cap without producing a final answer."""


class SupportsCheckpoint(Protocol):
    async def save(self, state: State) -> None: ...
    async def latest(self, run_id: UUID) -> State: ...


class SupportsTrace(Protocol):
    async def llm(
        self, state: State, comp: Completion, latency_ms: int, cost_usd: float
    ) -> None: ...
    async def tool(
        self, state: State, result: ToolResult, tool_name: str, latency_ms: int
    ) -> None: ...


class _NullCheckpointer:
    """Drops checkpoints. Used before the DB is wired in, and in unit tests."""

    async def save(self, state: State) -> None:
        return None

    async def latest(self, run_id: UUID) -> State:
        raise NotImplementedError("no checkpointer configured")


class _NullTracer:
    """Drops trace events."""

    async def llm(self, state: State, comp: Completion, latency_ms: int, cost_usd: float) -> None:
        return None

    async def tool(
        self, state: State, result: ToolResult, tool_name: str, latency_ms: int
    ) -> None:
        return None


class Agent:
    """Drives one run to completion.

    The only exceptions it raises are `BudgetExceeded` and `MaxStepsExceeded`; tool
    failures arrive as `ToolResult.error` and are fed back to the model as data. Every
    exit path persists `state.status`.
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: Registry,
        executor: Executor,
        checkpointer: SupportsCheckpoint | None = None,
        tracer: SupportsTrace | None = None,
        max_steps: int = 25,
        budget_usd: float = 1.00,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.checkpointer: SupportsCheckpoint = checkpointer or _NullCheckpointer()
        self.tracer: SupportsTrace = tracer or _NullTracer()
        self.max_steps = max_steps
        self.budget_usd = budget_usd
        self.total_cost_usd = 0.0

    async def run(self, state: State) -> str:
        """Run the loop until the model answers, the budget trips, or the cap is hit."""
        specs = self.registry.specs()

        while state.step_count < self.max_steps:
            t0 = time.perf_counter()
            comp = await self.llm.complete(state.messages, specs)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            step_cost = cost_usd(comp.prompt_tokens, comp.output_tokens)
            await self.tracer.llm(state, comp, latency_ms=latency_ms, cost_usd=step_cost)

            self.total_cost_usd += step_cost
            if self.total_cost_usd > self.budget_usd:
                state.status = "failed"
                await self.checkpointer.save(state)
                raise BudgetExceeded(
                    f"run cost ${self.total_cost_usd:.6f} exceeds budget "
                    f"${self.budget_usd:.6f}"
                )

            if not comp.tool_calls:
                state.messages.append(Message(role="assistant", content=comp.text))
                state.status = "completed"
                await self.checkpointer.save(state)
                return comp.text

            state.messages.append(
                Message(role="assistant", content=comp.text, tool_calls=comp.tool_calls)
            )
            t1 = time.perf_counter()
            results = await self.executor.run_all(comp.tool_calls)
            tool_latency_ms = int((time.perf_counter() - t1) * 1000)

            for result, call in zip(results, comp.tool_calls, strict=True):
                content = result.output if result.error is None else f"ERROR: {result.error}"
                state.messages.append(
                    Message(role="tool", content=content, tool_call_id=result.call_id)
                )
                await self.tracer.tool(
                    state, result, tool_name=call.name, latency_ms=tool_latency_ms
                )

            state.step_count += 1
            await self.checkpointer.save(state)

        state.status = "failed"
        await self.checkpointer.save(state)
        raise MaxStepsExceeded(f"run hit the {self.max_steps}-step cap without finishing")

    async def resume(self, run_id: UUID) -> str:
        """Load the newest checkpoint for `run_id` and continue the loop from there."""
        return await self.run(await self.checkpointer.latest(run_id))
