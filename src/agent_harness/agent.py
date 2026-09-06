"""The agent loop: model call, tool fan-out, checkpoint, repeat."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol
from uuid import UUID

from .context import CompactionResult, ContextManager
from .executor import Executor
from .llm import LLMClient, cost_usd
from .logging import get_logger, log_context
from .models import Completion, Message, State, ToolResult
from .sanitize import sanitize_tool_output
from .tools.base import Registry

logger = get_logger(__name__)


class BudgetExceeded(Exception):
    """The run's accumulated model spend passed the configured budget."""


class MaxStepsExceeded(Exception):
    """The run hit its iteration cap without producing a final answer."""


class RunPaused(Exception):
    """The run stopped cleanly at a step boundary because shutdown was requested.

    The checkpoint is coherent, so `agent resume <run_id>` continues from it.
    """


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
    async def spent_usd(self, run_id: UUID) -> float: ...


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

    async def spent_usd(self, run_id: UUID) -> float:
        return 0.0


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
        context: ContextManager | None = None,
        shutdown: asyncio.Event | None = None,
        sanitize: bool = True,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.checkpointer: SupportsCheckpoint = checkpointer or _NullCheckpointer()
        self.tracer: SupportsTrace = tracer or _NullTracer()
        self.max_steps = max_steps
        self.budget_usd = budget_usd
        self.context = context
        self._shutdown = shutdown
        self.sanitize = sanitize
        self.total_cost_usd = 0.0
        self.last_compaction: CompactionResult | None = None

    async def run(self, state: State) -> str:
        """Run the loop until the model answers, the budget trips, or the cap is hit."""
        specs = self.registry.specs()

        with log_context(run_id=str(state.run_id)):
            logger.info("run started", extra={"task": state.task[:200],
                                              "from_step": state.step_count})
            while state.step_count < self.max_steps:
                if self._shutdown is not None and self._shutdown.is_set():
                    return await self._pause(state)

                with log_context(step=state.step_count):
                    t0 = time.perf_counter()
                    comp = await self.llm.complete(self._prompt_messages(state), specs)
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    step_cost = cost_usd(comp.prompt_tokens, comp.output_tokens)
                    await self.tracer.llm(
                        state, comp, latency_ms=latency_ms, cost_usd=step_cost
                    )
                    logger.info(
                        "llm call",
                        extra={
                            "latency_ms": latency_ms,
                            "prompt_tokens": comp.prompt_tokens,
                            "output_tokens": comp.output_tokens,
                            "cost_usd": round(step_cost, 6),
                            "stop_reason": comp.stop_reason,
                            "tool_calls": len(comp.tool_calls),
                        },
                    )

                    self.total_cost_usd += step_cost
                    if self.total_cost_usd > self.budget_usd:
                        state.status = "failed"
                        await self.checkpointer.save(state)
                        logger.error(
                            "budget exceeded",
                            extra={"spent_usd": round(self.total_cost_usd, 6),
                                   "budget_usd": self.budget_usd},
                        )
                        raise BudgetExceeded(
                            f"run cost ${self.total_cost_usd:.6f} exceeds budget "
                            f"${self.budget_usd:.6f}"
                        )

                    if not comp.tool_calls:
                        state.messages.append(Message(role="assistant", content=comp.text))
                        state.status = "completed"
                        await self.checkpointer.save(state)
                        logger.info("run completed",
                                    extra={"steps": state.step_count,
                                           "cost_usd": round(self.total_cost_usd, 6)})
                        return comp.text

                    state.messages.append(
                        Message(role="assistant", content=comp.text,
                                tool_calls=comp.tool_calls)
                    )
                    t1 = time.perf_counter()
                    results = await self.executor.run_all(comp.tool_calls)
                    tool_latency_ms = int((time.perf_counter() - t1) * 1000)

                    for result, call in zip(results, comp.tool_calls, strict=True):
                        if result.error is not None:
                            content = f"ERROR: {result.error}"
                        else:
                            # Tool output is untrusted: redact credentials and flag any
                            # text trying to issue instructions before the model sees it.
                            clean = sanitize_tool_output(
                                result.output, call.name, redact=self.sanitize,
                                flag_injection=self.sanitize,
                            )
                            content = clean.text
                            if clean.modified:
                                logger.warning(
                                    "tool output sanitized",
                                    extra={"tool": call.name,
                                           "secrets_redacted": clean.secrets_redacted,
                                           "injections_found": clean.injections_found},
                                )
                                # Persist and replay the sanitized text, never the raw
                                # secret: state_json goes to the database too.
                                result = result.model_copy(update={"output": clean.text})
                        state.messages.append(
                            Message(role="tool", content=content,
                                    tool_call_id=result.call_id)
                        )
                        await self.tracer.tool(
                            state, result, tool_name=call.name, latency_ms=tool_latency_ms
                        )
                        logger.info(
                            "tool call",
                            extra={"tool": call.name, "call_id": result.call_id,
                                   "latency_ms": tool_latency_ms,
                                   "error": result.error},
                        )

                    state.step_count += 1
                    await self.checkpointer.save(state)

            state.status = "failed"
            await self.checkpointer.save(state)
            logger.error("max steps exceeded", extra={"max_steps": self.max_steps})
            raise MaxStepsExceeded(
                f"run hit the {self.max_steps}-step cap without finishing"
            )

    async def _pause(self, state: State) -> str:
        """Checkpoint and stop cleanly at a step boundary on shutdown request.

        Pausing between steps (rather than mid-tool) means the saved checkpoint is
        always a coherent state that `resume` can pick up, with no half-applied step.
        """
        state.status = "paused"
        await self.checkpointer.save(state)
        logger.warning("run paused for shutdown", extra={"step": state.step_count})
        raise RunPaused(
            f"run paused at step {state.step_count}; resume with: agent resume {state.run_id}"
        )

    def _prompt_messages(self, state: State) -> list[Message]:
        """The messages to send this iteration, compacted to fit the context window.

        Compaction applies to the *prompt*, never to `state.messages`: the checkpoint
        keeps the full history so replay and audit stay lossless, while the model sees a
        window that fits. `last_compaction` exposes what the most recent pass did.
        """
        if self.context is None:
            return state.messages
        result = self.context.compact(state.messages)
        self.last_compaction = result
        return result.messages

    async def resume(self, run_id: UUID) -> str:
        """Load the newest checkpoint for `run_id` and continue the loop from there.

        Seeds `total_cost_usd` from what the run has already spent, so the budget spans
        the whole run rather than resetting on every restart.
        """
        state = await self.checkpointer.latest(run_id)
        self.total_cost_usd = await self.tracer.spent_usd(run_id)
        return await self.run(state)
