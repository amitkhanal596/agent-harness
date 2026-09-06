"""spawn_agent: delegate a sub-task to a child agent run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from .base import Tool

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from ..agent import Agent


class AgentFactory(Protocol):
    """Builds a fresh Agent for a child run, with its own budget and step cap."""

    def __call__(self, max_steps: int, budget_usd: float) -> Agent: ...


class SpawnAgentInput(BaseModel):
    task: str = Field(min_length=1, description="The self-contained sub-task to delegate.")
    max_steps: int = Field(default=8, ge=1, le=25,
                           description="Iteration cap for the child run.")


class SpawnAgentTool(Tool):
    """Run a sub-task in a child agent with its own context, budget, and trace.

    Delegation exists to protect the parent's context window: a sub-task that reads ten
    files returns one summary to the parent instead of ten file bodies. The child is a
    real run — its own `agent_runs` row, checkpoints, and events — so `agent trace` and
    `agent replay` work on it, and the parent's true cost includes it via `run_trees`.

    Depth is capped and the child's budget is drawn from the parent's remaining
    allowance, so delegation cannot be used to escape the run budget or recurse forever.
    """

    name = "spawn_agent"
    description = (
        "Delegate a self-contained sub-task to a child agent, which runs its own "
        "tool-use loop and returns a single summarised answer. Use this to keep large "
        "intermediate work (reading many files, exploring a directory tree) out of your "
        "own context. Give it a complete, standalone instruction — it cannot see this "
        "conversation."
    )
    InputModel = SpawnAgentInput

    #: Depth beyond which delegation is refused. Two levels of delegation is plenty for
    #: decomposition; more usually signals a runaway loop.
    MAX_DEPTH = 2

    def __init__(
        self,
        factory: AgentFactory,
        checkpointer: object,
        parent_run_id: UUID,
        depth: int = 0,
        budget_usd: float = 0.25,
    ) -> None:
        self._factory = factory
        self._checkpointer = checkpointer
        self._parent_run_id = parent_run_id
        self._depth = depth
        self._budget_usd = budget_usd

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, SpawnAgentInput)
        if self._depth >= self.MAX_DEPTH:
            raise RuntimeError(
                f"delegation depth limit ({self.MAX_DEPTH}) reached; complete this "
                "sub-task directly instead of delegating further"
            )

        child_state = await self._checkpointer.create_run(  # type: ignore[attr-defined]
            args.task, parent_run_id=self._parent_run_id
        )
        child = self._factory(max_steps=args.max_steps, budget_usd=self._budget_usd)
        answer = await child.run(child_state)
        return f"[child run {child_state.run_id} completed]\n{answer}"
