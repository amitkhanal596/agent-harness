"""search_docs: semantic retrieval over the indexed document corpus."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..embeddings import VectorStore
from .base import Tool

MAX_RESULTS = 10


class SearchDocsInput(BaseModel):
    query: str = Field(description="What to search for, in natural language.")
    limit: int = Field(default=5, ge=1, le=MAX_RESULTS,
                       description="How many passages to return.")
    collection: str = Field(default="default", description="Corpus to search.")


class SearchDocsTool(Tool):
    """Retrieve passages semantically related to a query.

    Unlike `grep`, this matches on meaning rather than exact text, so it finds relevant
    passages that share no keywords with the query. Raises RuntimeError when the vector
    store is unavailable; the executor surfaces that to the model as a tool error.
    """

    name = "search_docs"
    description = (
        "Search the indexed document corpus for passages semantically related to a "
        "query. Use this when you need relevant context but do not know the exact "
        "wording to grep for. Returns passages with their source and similarity score."
    )
    InputModel = SearchDocsInput

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, SearchDocsInput)
        hits = await self._store.search(
            args.query, limit=args.limit, collection=args.collection
        )
        if not hits:
            return "[no matching passages]"
        blocks = [
            f"--- {hit['source']}#{hit['chunk_index']} "
            f"(similarity {hit['similarity']:.3f}) ---\n{hit['content']}"
            for hit in hits
        ]
        return "\n\n".join(blocks)
