"""§3.5: multi-agent orchestration — delegating a sub-task to a child run."""

from __future__ import annotations

from pathlib import Path

import asyncpg
import pytest

from agent_harness.agent import Agent
from agent_harness.checkpointer import Checkpointer
from agent_harness.executor import Executor
from agent_harness.models import Completion, ToolCall
from agent_harness.tools import build_registry
from agent_harness.tools.base import Registry
from agent_harness.tools.subagent import SpawnAgentInput, SpawnAgentTool
from agent_harness.tracer import Tracer
from helpers import MockLLMClient


def _child_factory(
    registry: Registry, script: list[Completion], checkpointer: Checkpointer,
    tracer: Tracer | None = None,
):
    def factory(max_steps: int, budget_usd: float) -> Agent:
        return Agent(
            llm=MockLLMClient(script),
            registry=registry,
            executor=Executor(registry, timeout=5.0, max_retries=0),
            checkpointer=checkpointer,
            tracer=tracer,
            max_steps=max_steps,
            budget_usd=budget_usd,
        )

    return factory


async def test_child_run_is_a_first_class_run(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> None:
    """A delegated sub-task gets its own row, checkpoints, and parent link."""
    cp = Checkpointer(db_pool)
    parent = await cp.create_run("parent task")
    registry = build_registry()

    tool = SpawnAgentTool(
        factory=_child_factory(
            registry,
            [Completion(stop_reason="end_turn", text="sub-task done")],
            cp,
        ),
        checkpointer=cp,
        parent_run_id=parent.run_id,
    )
    out = await tool.execute(SpawnAgentInput(task="do the sub-thing"))
    assert "sub-task done" in out

    children = await cp.children(parent.run_id)
    assert len(children) == 1
    assert children[0]["task"] == "do the sub-thing"
    assert children[0]["status"] == "completed"

    # The child is an ordinary run: replay works on it unchanged.
    child_id = children[0]["id"]
    assert await cp.list_steps(child_id)


async def test_child_uses_its_own_tools_and_returns_a_summary(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> None:
    """Delegation keeps bulky intermediate work out of the parent's context."""
    cp = Checkpointer(db_pool)
    parent = await cp.create_run("parent")
    registry = build_registry()

    child_script = [
        Completion(
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(id="c1", name="read_file",
                         input={"path": str(temp_repo / "config.txt")})
            ],
        ),
        Completion(stop_reason="end_turn", text="the port is 8080"),
    ]
    tool = SpawnAgentTool(
        factory=_child_factory(registry, child_script, cp),
        checkpointer=cp,
        parent_run_id=parent.run_id,
    )
    out = await tool.execute(SpawnAgentInput(task="find the port"))

    # The parent sees the summary, not the raw file contents.
    assert "the port is 8080" in out
    assert "debug=false" not in out


async def test_depth_limit_prevents_runaway_recursion(
    db_pool: asyncpg.Pool
) -> None:
    cp = Checkpointer(db_pool)
    parent = await cp.create_run("parent")
    registry = build_registry()
    tool = SpawnAgentTool(
        factory=_child_factory(registry, [Completion(stop_reason="end_turn", text="x")], cp),
        checkpointer=cp,
        parent_run_id=parent.run_id,
        depth=SpawnAgentTool.MAX_DEPTH,
    )
    with pytest.raises(RuntimeError, match="depth limit"):
        await tool.execute(SpawnAgentInput(task="go deeper"))


async def test_child_budget_is_bounded(db_pool: asyncpg.Pool) -> None:
    """A child cannot spend beyond the allowance it was given."""
    from agent_harness.agent import BudgetExceeded

    cp = Checkpointer(db_pool)
    parent = await cp.create_run("parent")
    registry = build_registry()
    expensive = [
        Completion(stop_reason="end_turn", text="pricey",
                   prompt_tokens=1_000_000, output_tokens=0)
    ]
    tool = SpawnAgentTool(
        factory=_child_factory(registry, expensive, cp),
        checkpointer=cp,
        parent_run_id=parent.run_id,
        budget_usd=0.01,
    )
    with pytest.raises(BudgetExceeded):
        await tool.execute(SpawnAgentInput(task="expensive sub-task"))


async def test_tree_cost_rolls_up_children(
    db_pool: asyncpg.Pool
) -> None:
    """A parent's true cost includes what it delegated."""
    cp, tracer = Checkpointer(db_pool), Tracer(db_pool)
    parent = await cp.create_run("parent")
    registry = build_registry()

    tool = SpawnAgentTool(
        factory=_child_factory(
            registry,
            [Completion(stop_reason="end_turn", text="done",
                        prompt_tokens=1000, output_tokens=100)],
            cp,
            tracer,
        ),
        checkpointer=cp,
        parent_run_id=parent.run_id,
    )
    await tool.execute(SpawnAgentInput(task="sub"))

    tree = await cp.tree_cost(parent.run_id)
    assert tree["run_count"] == 2  # parent + child
    assert tree["max_depth"] == 1
    assert float(tree["tree_cost_usd"]) > 0  # the child's spend is included


async def test_parent_loop_can_call_the_spawn_tool(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> None:
    """End to end: a parent agent delegates through the normal tool-call path."""
    cp = Checkpointer(db_pool)
    parent_state = await cp.create_run("investigate")
    registry = build_registry()

    child_registry = build_registry()
    spawn = SpawnAgentTool(
        factory=_child_factory(
            child_registry,
            [Completion(stop_reason="end_turn", text="child found the answer: 8080")],
            cp,
        ),
        checkpointer=cp,
        parent_run_id=parent_state.run_id,
    )
    registry.register(spawn)

    parent = Agent(
        llm=MockLLMClient(
            [
                Completion(
                    stop_reason="tool_use",
                    tool_calls=[ToolCall(id="s1", name="spawn_agent",
                                         input={"task": "find the port"})],
                ),
                Completion(stop_reason="end_turn", text="The port is 8080."),
            ]
        ),
        registry=registry,
        executor=Executor(registry, timeout=10.0, max_retries=0),
        checkpointer=cp,
        max_steps=5,
    )
    assert await parent.run(parent_state) == "The port is 8080."

    children = await cp.children(parent_state.run_id)
    assert len(children) == 1
    tool_msg = next(m for m in parent_state.messages if m.role == "tool")
    assert "child run" in tool_msg.content


def test_spawn_input_bounds_max_steps() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SpawnAgentInput(task="x", max_steps=0)
    with pytest.raises(ValidationError):
        SpawnAgentInput(task="x", max_steps=999)
    with pytest.raises(ValidationError):
        SpawnAgentInput(task="")


def test_spawn_tool_is_registered_by_name() -> None:
    assert SpawnAgentTool.name == "spawn_agent"
    assert "self-contained" in SpawnAgentTool.description
