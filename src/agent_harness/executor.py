"""Tool execution: schema validation, timeouts, retries, and concurrent fan-out."""

from __future__ import annotations

import asyncio
import logging

from pydantic import ValidationError

from .models import ToolCall, ToolResult
from .tools.base import Registry

logger = logging.getLogger(__name__)

BACKOFF_BASE_SECONDS = 0.1

#: Exceptions worth retrying: transient I/O and timeouts. Everything else is fatal to
#: the individual call (but still returned as data, never raised into the loop).
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
)


class FatalToolError(Exception):
    """Raised by a tool to abort the whole step: siblings are cancelled (M6)."""


def is_transient(exc: BaseException) -> bool:
    """Whether `exc` is worth retrying. FatalToolError is never transient."""
    if isinstance(exc, FatalToolError):
        return False
    return isinstance(exc, TRANSIENT_ERRORS)


class Executor:
    """Runs tool calls with validation, per-call timeout, and bounded retries.

    `run` never raises for a tool failure — errors come back as `ToolResult.error`. The
    single exception is `FatalToolError`, which `run_all` propagates so siblings are
    cancelled.
    """

    def __init__(self, registry: Registry, timeout: float, max_retries: int) -> None:
        self._registry = registry
        self._timeout = timeout
        self._max_retries = max_retries

    async def run(self, call: ToolCall) -> ToolResult:
        """Execute one tool call, converting every failure into `ToolResult.error`."""
        tool = self._registry.get(call.name)
        if tool is None:
            return ToolResult(call_id=call.id, error=f"unknown tool: {call.name!r}")

        try:
            args = tool.InputModel.model_validate(call.input)
        except ValidationError as exc:
            return ToolResult(call_id=call.id, error=f"invalid input for {call.name}: {exc}")

        last_error: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                output = await asyncio.wait_for(tool.execute(args), timeout=self._timeout)
                return ToolResult(call_id=call.id, output=output)
            except FatalToolError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - failures are data, not crashes
                last_error = exc
                if not is_transient(exc) or attempt == self._max_retries:
                    break
                delay = BACKOFF_BASE_SECONDS * (2**attempt)
                logger.debug("retrying %s after %s (attempt %d)", call.name, exc, attempt + 1)
                await asyncio.sleep(delay)

        return ToolResult(call_id=call.id, error=_describe(call.name, last_error))

    async def run_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run every call concurrently, returning results in the *input* order.

        A `FatalToolError` from any call cancels its siblings; the calls that had already
        finished keep their results and the cancelled ones report cancellation.
        """
        if not calls:
            return []

        results: list[ToolResult | None] = [None] * len(calls)

        async def worker(index: int, call: ToolCall) -> None:
            results[index] = await self.run(call)

        fatal: BaseException | None = None
        try:
            async with asyncio.TaskGroup() as group:
                for index, call in enumerate(calls):
                    group.create_task(worker(index, call))
        except* FatalToolError as group_exc:
            fatal = group_exc.exceptions[0]

        return [
            result
            if result is not None
            else ToolResult(
                call_id=calls[i].id,
                error=(
                    f"cancelled: sibling tool raised {_describe_exc(fatal)}"
                    if fatal is not None
                    else "cancelled"
                ),
            )
            for i, result in enumerate(results)
        ]


def _describe_exc(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown error"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _describe(tool_name: str, exc: BaseException | None) -> str:
    if isinstance(exc, TimeoutError):
        return f"{tool_name} timed out"
    return f"{tool_name} failed: {_describe_exc(exc)}"
