"""Context-window management: keep the conversation inside the model's token budget.

The loop resends the whole message list on every iteration, so an unmanaged run grows
quadratically in tokens and eventually exceeds the model's context window. This module
decides what to keep verbatim, what to shrink, and what to drop, without ever breaking
the structural invariant the provider requires: **every `tool_use` block must still be
answered by a `tool_result` with the same id, in the same order**.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Message

#: Rough bytes-per-token for English text plus code. The exact ratio varies by
#: tokenizer; this is deliberately conservative (under-estimating tokens would let the
#: window overflow, which is the failure we are preventing).
BYTES_PER_TOKEN = 3.5

#: Marker left in place of content removed by compaction, so the model can tell the
#: difference between "this tool returned nothing" and "this was trimmed to fit".
ELISION = "[...trimmed to fit the context window...]"


def estimate_tokens(text: str) -> int:
    """Approximate token count for `text`.

    A heuristic, not a tokenizer: the harness must not take a tokenizer dependency per
    provider, and the number is only used to decide when to compact. Rounds up so a
    short string still costs at least one token.
    """
    if not text:
        return 0
    return max(1, int(len(text) / BYTES_PER_TOKEN) + 1)


def message_tokens(message: Message) -> int:
    """Approximate tokens for one message, including its tool-call payloads."""
    total = estimate_tokens(message.content)
    for call in message.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(str(call.input))
    return total


def conversation_tokens(messages: list[Message]) -> int:
    """Approximate tokens for a whole conversation."""
    return sum(message_tokens(m) for m in messages)


@dataclass(frozen=True)
class CompactionResult:
    """What a compaction pass did, for tracing and tests."""

    messages: list[Message]
    tokens_before: int
    tokens_after: int
    truncated_tool_results: int
    dropped_messages: int

    @property
    def changed(self) -> bool:
        return self.truncated_tool_results > 0 or self.dropped_messages > 0


class ContextManager:
    """Keeps a conversation under `max_tokens` using progressively lossier passes.

    Strategy, in order, stopping as soon as the conversation fits:

    1. **Truncate old tool results.** Tool output is the bulk of a long run and the
       least valuable to keep verbatim once the model has acted on it. The most recent
       `keep_recent_steps` worth of messages are never touched, so the model always sees
       full fidelity for what it is currently working on.
    2. **Drop the oldest turns entirely.** If truncation is not enough, whole
       assistant/tool groups are dropped from the front — never the first user message
       (the task), and never a partial group, because orphaning a `tool_use` without its
       `tool_result` makes the request invalid.

    The task message and the recent window are always preserved, so a compacted
    conversation is still coherent: the model knows what it was asked and what it just
    did.
    """

    def __init__(
        self,
        max_tokens: int = 150_000,
        keep_recent_messages: int = 8,
        truncated_result_chars: int = 2_000,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if keep_recent_messages < 1:
            raise ValueError("keep_recent_messages must be positive")
        self.max_tokens = max_tokens
        self.keep_recent_messages = keep_recent_messages
        self.truncated_result_chars = truncated_result_chars

    def compact(self, messages: list[Message]) -> CompactionResult:
        """Return a copy of `messages` that fits the window, plus what it cost."""
        before = conversation_tokens(messages)
        if before <= self.max_tokens or not messages:
            return CompactionResult(list(messages), before, before, 0, 0)

        working = [m.model_copy(deep=True) for m in messages]
        truncated = self._truncate_old_tool_results(working)

        dropped = 0
        if conversation_tokens(working) > self.max_tokens:
            working, dropped = self._drop_oldest_groups(working)

        return CompactionResult(
            messages=working,
            tokens_before=before,
            tokens_after=conversation_tokens(working),
            truncated_tool_results=truncated,
            dropped_messages=dropped,
        )

    def _protected_from(self, messages: list[Message]) -> int:
        """Index from which messages are never modified (the recent window)."""
        return max(0, len(messages) - self.keep_recent_messages)

    def _truncate_old_tool_results(self, messages: list[Message]) -> int:
        """Shorten tool-result content outside the recent window. Returns the count."""
        protected = self._protected_from(messages)
        count = 0
        for index, message in enumerate(messages):
            if index >= protected or message.role != "tool":
                continue
            if len(message.content) <= self.truncated_result_chars:
                continue
            message.content = message.content[: self.truncated_result_chars] + f"\n{ELISION}"
            count += 1
        return count

    def _drop_oldest_groups(self, messages: list[Message]) -> tuple[list[Message], int]:
        """Drop whole turn-groups from the front until the conversation fits.

        A "group" is an assistant message plus the tool messages answering it, so the
        `tool_use`/`tool_result` pairing is never broken. The leading user message (the
        task) and the recent window always survive.
        """
        if not messages:
            return messages, 0

        head = messages[0] if messages[0].role == "user" else None
        body_start = 1 if head is not None else 0
        protected = self._protected_from(messages)

        # Group boundaries in the droppable region: each assistant message starts a group.
        droppable = messages[body_start:protected]
        keep_tail = messages[protected:]

        dropped = 0
        while droppable and conversation_tokens(
            ([head] if head else []) + droppable + keep_tail
        ) > self.max_tokens:
            # Drop the first group: one leading message plus any tool replies after it.
            removed = 1
            while removed < len(droppable) and droppable[removed].role == "tool":
                removed += 1
            droppable = droppable[removed:]
            dropped += removed

        return ([head] if head else []) + droppable + keep_tail, dropped
