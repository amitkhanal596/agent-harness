"""Embedding providers and the pgvector-backed document store.

Retrieval lives in the same Postgres instance as runs and checkpoints, so it adds no new
infrastructure and a chunk's embedding is written transactionally beside its text.

`EmbeddingClient` mirrors `LLMClient`: a Protocol with a real implementation and a
deterministic test double, so the store is testable without network access.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any, Protocol

import asyncpg
import httpx

#: Must match the `vector(N)` column width in `003_vector_store.sql`.
EMBEDDING_DIM = 1024

#: Chunking defaults. Overlap keeps a sentence that straddles a boundary retrievable
#: from either side.
CHUNK_CHARS = 1_200
CHUNK_OVERLAP = 150


class EmbeddingClient(Protocol):
    """Turns text into fixed-width vectors."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class VoyageEmbeddingClient:
    """Anthropic's recommended embedding provider (Voyage AI).

    Network errors propagate; the caller decides whether to retry. Batches are capped
    because the API rejects oversized requests.
    """

    MAX_BATCH = 128

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3",
        base_url: str = "https://api.voyageai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts`, preserving order."""
        if not texts:
            return []
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for start in range(0, len(texts), self.MAX_BATCH):
                batch = texts[start : start + self.MAX_BATCH]
                resp = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"input": batch, "model": self._model},
                )
                resp.raise_for_status()
                payload = resp.json()
                # The API documents ordered results, but it also returns an index per
                # item; sorting on it is cheap insurance against silent misalignment.
                items = sorted(payload["data"], key=lambda d: d["index"])
                out.extend(item["embedding"] for item in items)
        return out


class HashEmbeddingClient:
    """Deterministic local embeddings: hashed bag-of-words, L2-normalised.

    Not semantic — it cannot match synonyms — but it is stable, offline, and free, which
    makes the retrieval *plumbing* (chunking, storage, cosine ranking, SQL) testable
    without an API key. Lexical overlap still ranks sensibly, which is enough to assert
    that the right chunk comes back for a keyword query.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9_]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = struct.unpack("<Q", digest)[0] % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # An all-zero vector has undefined cosine distance; use a fixed unit vector.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def chunk_text(
    text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split `text` into overlapping chunks, preferring paragraph/line boundaries.

    Breaking mid-sentence loses the context that makes a chunk answerable, so the split
    point backs up to the last blank line or newline within the window when there is one.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be >= 0 and < chunk_chars")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            window = text[start:end]
            boundary = max(window.rfind("\n\n"), window.rfind("\n"))
            # Only honour a boundary in the back half, else chunks get uselessly small.
            if boundary > chunk_chars // 2:
                end = start + boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def to_pgvector(values: list[float]) -> str:
    """Render a Python list as a pgvector literal.

    asyncpg has no native codec for `vector`, so it is passed as text and cast in SQL.
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


class VectorStore:
    """Chunk, embed, and retrieve documents from the `documents` table.

    Raises `RuntimeError` when pgvector is not installed on the server, so the failure
    is a clear message rather than a confusing missing-table error.
    """

    def __init__(self, pool: asyncpg.Pool, embedder: EmbeddingClient) -> None:
        self._pool = pool
        self._embedder = embedder

    async def available(self) -> bool:
        """Whether the server has pgvector and the documents table exists."""
        return bool(
            await self._pool.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables"
                " WHERE table_name = 'documents')"
            )
        )

    async def _require(self) -> None:
        if not await self.available():
            raise RuntimeError(
                "vector store unavailable: pgvector is not installed on this server. "
                "Install it (e.g. the pgvector/pgvector Docker image) and re-run "
                "`agent migrate`."
            )

    async def index_text(
        self,
        source: str,
        text: str,
        collection: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunk, embed, and store `text`. Re-indexing a source replaces its chunks."""
        await self._require()
        chunks = chunk_text(text)
        if not chunks:
            return 0
        vectors = await self._embedder.embed(chunks)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Replace rather than append, so re-indexing an edited file does not
                # leave stale chunks that still match queries.
                await conn.execute(
                    "DELETE FROM documents WHERE collection = $1 AND source = $2",
                    collection,
                    source,
                )
                for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                    await conn.execute(
                        """
                        INSERT INTO documents
                            (collection, source, chunk_index, content, embedding, metadata)
                        VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
                        """,
                        collection,
                        source,
                        index,
                        chunk,
                        to_pgvector(vector),
                        json.dumps(metadata or {}),
                    )
        return len(chunks)

    async def search(
        self, query: str, limit: int = 5, collection: str = "default"
    ) -> list[dict[str, Any]]:
        """Return the `limit` chunks closest to `query` by cosine similarity."""
        await self._require()
        vectors = await self._embedder.embed([query])
        rows = await self._pool.fetch(
            """
            SELECT source, chunk_index, content, metadata,
                   1 - (embedding <=> $1::vector) AS similarity
              FROM documents
             WHERE collection = $2
             ORDER BY embedding <=> $1::vector
             LIMIT $3
            """,
            to_pgvector(vectors[0]),
            collection,
            limit,
        )
        return [dict(row) for row in rows]

    async def count(self, collection: str = "default") -> int:
        """Number of stored chunks in `collection`."""
        await self._require()
        total = await self._pool.fetchval(
            "SELECT count(*) FROM documents WHERE collection = $1", collection
        )
        return int(total or 0)

    async def clear(self, collection: str = "default") -> int:
        """Delete every chunk in `collection`; returns how many were removed."""
        await self._require()
        result = await self._pool.execute(
            "DELETE FROM documents WHERE collection = $1", collection
        )
        return int(result.split()[-1]) if result.startswith("DELETE") else 0
