"""§2.1: context-window management — compaction that fits without breaking the protocol."""

from __future__ import annotations

from uuid import uuid4

import pytest

from agent_harness.agent import Agent
from agent_harness.context import (
    ELISION,
    ContextManager,
    conversation_tokens,
    estimate_tokens,
    message_tokens,
)
from agent_harness.executor import Executor
from agent_harness.llm import to_anthropic_messages
from agent_harness.models import Completion, Message, State, ToolCall
from agent_harness.tools import build_registry
from helpers import MockLLMClient


def _conversation(steps: int, result_size: int = 5_000) -> list[Message]:
    """A task message followed by `steps` assistant/tool turn-groups."""
    messages = [Message(role="user", content="find the bug")]
    for i in range(steps):
        messages.append(
            Message(
                role="assistant",
                content=f"step {i}",
                tool_calls=[ToolCall(id=f"t{i}", name="read_file", input={"path": f"f{i}"})],
            )
        )
        messages.append(Message(role="tool", tool_call_id=f"t{i}", content="x" * result_size))
    return messages


def _tool_use_ids(messages: list[Message]) -> list[str]:
    return [c.id for m in messages if m.role == "assistant" for c in m.tool_calls]


def _tool_result_ids(messages: list[Message]) -> list[str]:
    return [m.tool_call_id for m in messages if m.role == "tool"]


# --- estimation ---------------------------------------------------------------------


def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") >= 1
    assert estimate_tokens("a" * 3500) > estimate_tokens("a" * 350)


def test_message_tokens_includes_tool_call_payload() -> None:
    plain = Message(role="assistant", content="hi")
    with_call = Message(
        role="assistant",
        content="hi",
        tool_calls=[ToolCall(id="t", name="read_file", input={"path": "/very/long/path.txt"})],
    )
    assert message_tokens(with_call) > message_tokens(plain)


def test_conversation_tokens_sums_messages() -> None:
    messages = _conversation(3)
    assert conversation_tokens(messages) == sum(message_tokens(m) for m in messages)


# --- no-op path ---------------------------------------------------------------------


def test_short_conversation_is_untouched() -> None:
    messages = _conversation(2, result_size=50)
    result = ContextManager(max_tokens=1_000_000).compact(messages)
    assert result.messages == messages
    assert not result.changed
    assert result.tokens_before == result.tokens_after


def test_empty_conversation_is_safe() -> None:
    result = ContextManager(max_tokens=10).compact([])
    assert result.messages == []
    assert not result.changed


# --- truncation pass ----------------------------------------------------------------


def test_old_tool_results_are_truncated_before_anything_is_dropped() -> None:
    messages = _conversation(10, result_size=20_000)
    result = ContextManager(
        max_tokens=20_000, keep_recent_messages=4, truncated_result_chars=500
    ).compact(messages)

    assert result.truncated_tool_results > 0
    assert result.tokens_after < result.tokens_before
    trimmed = [m for m in result.messages if m.role == "tool" and ELISION in m.content]
    assert trimmed, "expected at least one elided tool result"
    assert all(len(m.content) < 20_000 for m in trimmed)


def test_recent_messages_keep_full_fidelity() -> None:
    """The model must see untruncated output for what it is currently working on."""
    messages = _conversation(10, result_size=20_000)
    result = ContextManager(
        max_tokens=20_000, keep_recent_messages=4, truncated_result_chars=500
    ).compact(messages)

    for message in result.messages[-4:]:
        assert ELISION not in message.content


def test_compaction_does_not_mutate_the_input() -> None:
    """State stays lossless: compaction works on a copy, so checkpoints keep everything."""
    messages = _conversation(10, result_size=20_000)
    original = [m.model_copy(deep=True) for m in messages]
    ContextManager(max_tokens=5_000, truncated_result_chars=100).compact(messages)
    assert messages == original


# --- dropping pass ------------------------------------------------------------------


def test_oldest_groups_are_dropped_when_truncation_is_not_enough() -> None:
    messages = _conversation(30, result_size=8_000)
    result = ContextManager(
        max_tokens=4_000, keep_recent_messages=4, truncated_result_chars=200
    ).compact(messages)

    assert result.dropped_messages > 0
    assert len(result.messages) < len(messages)
    assert result.tokens_after <= result.tokens_before


def test_the_task_message_always_survives() -> None:
    """Drop everything else if you must, but the model must still know the task."""
    messages = _conversation(40, result_size=10_000)
    result = ContextManager(
        max_tokens=500, keep_recent_messages=2, truncated_result_chars=50
    ).compact(messages)

    assert result.messages[0].role == "user"
    assert result.messages[0].content == "find the bug"


# --- the invariant that actually matters --------------------------------------------


@pytest.mark.parametrize("max_tokens", [200, 1_000, 4_000, 20_000])
@pytest.mark.parametrize("steps", [5, 20, 50])
def test_every_tool_use_still_has_its_tool_result(max_tokens: int, steps: int) -> None:
    """Orphaning a tool_use from its tool_result makes the API request invalid.

    This is the invariant compaction is most likely to break, so it is checked across a
    grid of window sizes and conversation lengths.
    """
    messages = _conversation(steps, result_size=6_000)
    result = ContextManager(
        max_tokens=max_tokens, keep_recent_messages=4, truncated_result_chars=200
    ).compact(messages)

    assert _tool_use_ids(result.messages) == _tool_result_ids(result.messages)


@pytest.mark.parametrize("steps", [3, 15, 40])
def test_compacted_conversation_still_maps_to_valid_provider_messages(steps: int) -> None:
    """End-to-end guard: the compacted list must survive the Anthropic mapping."""
    messages = _conversation(steps, result_size=9_000)
    result = ContextManager(
        max_tokens=2_000, keep_recent_messages=4, truncated_result_chars=200
    ).compact(messages)

    mapped = to_anthropic_messages(result.messages)
    tool_use_ids = [
        block["id"]
        for m in mapped
        if m["role"] == "assistant"
        for block in m["content"]
        if block["type"] == "tool_use"
    ]
    tool_result_ids = [
        block["tool_use_id"]
        for m in mapped
        if m["role"] == "user"
        for block in m["content"]
        if block["type"] == "tool_result"
    ]
    assert tool_use_ids == tool_result_ids


def test_ordering_is_preserved() -> None:
    messages = _conversation(20, result_size=6_000)
    result = ContextManager(
        max_tokens=3_000, keep_recent_messages=4, truncated_result_chars=200
    ).compact(messages)
    ids = _tool_use_ids(result.messages)
    assert ids == sorted(ids, key=lambda i: int(i[1:]))


# --- configuration ------------------------------------------------------------------


def test_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError):
        ContextManager(max_tokens=0)
    with pytest.raises(ValueError):
        ContextManager(keep_recent_messages=0)


# --- integration with the loop ------------------------------------------------------


async def test_loop_sends_compacted_messages_but_checkpoints_everything() -> None:
    """The prompt shrinks; the persisted state stays complete."""
    registry = build_registry()
    llm = MockLLMClient(
        [
            Completion(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="list_dir", input={"path": "."})],
            ),
            Completion(stop_reason="end_turn", text="done"),
        ]
    )
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        max_steps=5,
        context=ContextManager(max_tokens=20, keep_recent_messages=1,
                               truncated_result_chars=10),
    )
    state = State(
        run_id=uuid4(),
        task="t",
        messages=[Message(role="user", content="x" * 5000)],
    )
    assert await agent.run(state) == "done"

    # The model saw a compacted prompt on the second call...
    assert agent.last_compaction is not None
    assert agent.last_compaction.changed
    # ...but the full history is still in the state that gets checkpointed.
    assert any(len(m.content) >= 5000 for m in state.messages)


async def test_loop_without_a_context_manager_is_unchanged() -> None:
    """Compaction is opt-in: no manager means the old pass-through behavior."""
    registry = build_registry()
    llm = MockLLMClient([Completion(stop_reason="end_turn", text="ok")])
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        max_steps=5,
    )
    state = State(run_id=uuid4(), task="t", messages=[Message(role="user", content="hi")])
    assert await agent.run(state) == "ok"
    assert agent.last_compaction is None
