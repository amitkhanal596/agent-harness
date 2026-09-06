"""Test doubles shared across the suite."""

from __future__ import annotations

from agent_harness.models import Completion, Message, ToolSpec


class MockLLMClient:
    """Replays a scripted list of Completions; records what it was asked.

    Raises RuntimeError when the script is exhausted so runaway loops fail loudly.
    Set `repeat_last=True` to return the final Completion forever (max-steps tests).
    """

    def __init__(self, script: list[Completion], repeat_last: bool = False) -> None:
        self.script = list(script)
        self.repeat_last = repeat_last
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[ToolSpec]] = []

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        self.calls.append([m.model_copy(deep=True) for m in messages])
        self.tools_seen.append(list(tools))
        if not self.script:
            raise RuntimeError("MockLLMClient script exhausted")
        if self.repeat_last and len(self.script) == 1:
            return self.script[0]
        return self.script.pop(0)
