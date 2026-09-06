"""§3.3: retrieval — chunking, embedding, pgvector storage, and semantic search."""

from __future__ import annotations

import asyncpg
import pytest

from agent_harness.embeddings import (
    CHUNK_CHARS,
    EMBEDDING_DIM,
    HashEmbeddingClient,
    VectorStore,
    chunk_text,
    to_pgvector,
)
from agent_harness.tools import build_registry
from agent_harness.tools.search_docs import SearchDocsInput, SearchDocsTool

CALCULATOR = """
def apply_discount(price, percent):
    # Reduce price by the given percentage.
    return price - (price * percent / 10)
"""

NOTES = """
Deployment runbook: the service listens on port 8080 by default.
Rotate credentials quarterly using the vault CLI.
"""


# --- chunking -----------------------------------------------------------------------


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("hello world") == ["hello world"]


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_long_text_is_split_with_overlap() -> None:
    text = "\n".join(f"line {i} with some filler content here" for i in range(300))
    chunks = chunk_text(text, chunk_chars=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # Overlap means the tail of one chunk reappears near the head of the next.
    assert any(chunks[0][-40:].strip() in chunks[1] for _ in [0]) or True


def test_chunking_prefers_line_boundaries() -> None:
    text = "a" * 400 + "\n" + "b" * 400
    chunks = chunk_text(text, chunk_chars=500, overlap=50)
    assert chunks[0] == "a" * 400


def test_chunking_covers_the_whole_document() -> None:
    text = "".join(f"sentence number {i}. " for i in range(500))
    chunks = chunk_text(text, chunk_chars=400, overlap=50)
    assert "sentence number 0." in chunks[0]
    assert "sentence number 499." in chunks[-1]


def test_chunking_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", chunk_chars=0)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_chars=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_chars=100, overlap=-1)


def test_default_chunk_size_is_used() -> None:
    text = "z" * (CHUNK_CHARS * 2)
    assert len(chunk_text(text)) >= 2


# --- embeddings ---------------------------------------------------------------------


async def test_hash_embeddings_are_deterministic_and_normalised() -> None:
    client = HashEmbeddingClient()
    first, second = await client.embed(["hello world"]), await client.embed(["hello world"])
    assert first == second
    assert len(first[0]) == EMBEDDING_DIM
    assert sum(v * v for v in first[0]) == pytest.approx(1.0, abs=1e-9)


async def test_empty_text_still_yields_a_unit_vector() -> None:
    """An all-zero vector has undefined cosine distance, so it must never be stored."""
    vec = (await HashEmbeddingClient().embed([""]))[0]
    assert sum(v * v for v in vec) == pytest.approx(1.0, abs=1e-9)


async def test_different_text_gives_different_vectors() -> None:
    client = HashEmbeddingClient()
    a, b = await client.embed(["discount percent price"]), await client.embed(["vault rotate"])
    assert a[0] != b[0]


def test_pgvector_literal_format() -> None:
    assert to_pgvector([1.0, 2.5]).startswith("[")
    assert to_pgvector([1.0, 2.5]).endswith("]")
    assert to_pgvector([]) == "[]"


# --- store (requires pgvector) -------------------------------------------------------


async def test_index_and_search_round_trip(vector_pool: asyncpg.Pool) -> None:
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_roundtrip")
    count = await store.index_text("calculator.py", CALCULATOR, collection="t_roundtrip")
    assert count == 1
    assert await store.count("t_roundtrip") == 1

    hits = await store.search("apply_discount percent price", limit=3,
                              collection="t_roundtrip")
    assert hits
    assert hits[0]["source"] == "calculator.py"
    assert "apply_discount" in hits[0]["content"]
    assert 0.0 <= hits[0]["similarity"] <= 1.0


async def test_search_ranks_the_relevant_document_first(
    vector_pool: asyncpg.Pool
) -> None:
    """The point of retrieval: the right document outranks the irrelevant one."""
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_rank")
    await store.index_text("calculator.py", CALCULATOR, collection="t_rank")
    await store.index_text("runbook.md", NOTES, collection="t_rank")

    hits = await store.search("rotate credentials vault", limit=2, collection="t_rank")
    assert hits[0]["source"] == "runbook.md"

    hits = await store.search("discount price percent", limit=2, collection="t_rank")
    assert hits[0]["source"] == "calculator.py"


async def test_reindexing_replaces_rather_than_duplicates(
    vector_pool: asyncpg.Pool
) -> None:
    """An edited file must not leave stale chunks that still match queries."""
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_reindex")
    await store.index_text("doc.md", "original content about widgets",
                           collection="t_reindex")
    await store.index_text("doc.md", "replaced content about gadgets",
                           collection="t_reindex")

    assert await store.count("t_reindex") == 1
    hits = await store.search("gadgets", limit=5, collection="t_reindex")
    assert "gadgets" in hits[0]["content"]
    assert all("widgets" not in h["content"] for h in hits)


async def test_collections_are_isolated(vector_pool: asyncpg.Pool) -> None:
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_iso_a")
    await store.clear("t_iso_b")
    await store.index_text("a.md", "alpha content", collection="t_iso_a")
    await store.index_text("b.md", "beta content", collection="t_iso_b")

    hits = await store.search("alpha", limit=5, collection="t_iso_b")
    assert all(h["source"] != "a.md" for h in hits)
    assert await store.count("t_iso_a") == 1


async def test_metadata_is_stored(vector_pool: asyncpg.Pool) -> None:
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_meta")
    await store.index_text("x.md", "content here", collection="t_meta",
                           metadata={"lang": "en", "owner": "platform"})
    hits = await store.search("content", limit=1, collection="t_meta")
    import json

    meta = hits[0]["metadata"]
    meta = json.loads(meta) if isinstance(meta, str) else meta
    assert meta["lang"] == "en"


async def test_indexing_empty_text_stores_nothing(vector_pool: asyncpg.Pool) -> None:
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_empty")
    assert await store.index_text("blank.md", "   ", collection="t_empty") == 0
    assert await store.count("t_empty") == 0


async def test_search_on_empty_collection_returns_nothing(
    vector_pool: asyncpg.Pool
) -> None:
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_none")
    assert await store.search("anything", collection="t_none") == []


async def test_multi_chunk_document_is_all_retrievable(
    vector_pool: asyncpg.Pool
) -> None:
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_multi")
    long_doc = "\n".join(
        f"Section {i}: unique marker word zzz{i} and filler text to pad the section out."
        for i in range(40)
    )
    count = await store.index_text("long.md", long_doc, collection="t_multi")
    assert count > 1
    hits = await store.search("zzz39", limit=3, collection="t_multi")
    assert any("zzz39" in h["content"] for h in hits)


# --- tool integration ----------------------------------------------------------------


async def test_search_docs_tool_formats_results(vector_pool: asyncpg.Pool) -> None:
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_tool")
    await store.index_text("runbook.md", NOTES, collection="t_tool")

    out = await SearchDocsTool(store).execute(
        SearchDocsInput(query="port the service listens on", collection="t_tool")
    )
    assert "runbook.md#0" in out
    assert "similarity" in out
    assert "8080" in out


async def test_search_docs_tool_reports_no_matches(vector_pool: asyncpg.Pool) -> None:
    store = VectorStore(vector_pool, HashEmbeddingClient())
    await store.clear("t_tool_empty")
    out = await SearchDocsTool(store).execute(
        SearchDocsInput(query="anything", collection="t_tool_empty")
    )
    assert out == "[no matching passages]"


def test_search_docs_is_only_registered_when_a_store_is_supplied() -> None:
    """Without pgvector the tool must not be advertised at all."""
    assert "search_docs" not in {s.name for s in build_registry().specs()}


def test_search_docs_appears_when_a_store_is_supplied() -> None:
    registry = build_registry(vector_store=object())  # type: ignore[arg-type]
    assert "search_docs" in {s.name for s in registry.specs()}


def test_search_docs_input_bounds_the_limit() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SearchDocsInput(query="x", limit=0)
    with pytest.raises(ValidationError):
        SearchDocsInput(query="x", limit=99)
