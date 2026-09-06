"""grep: search a file or directory tree for a regex, returning numbered matches."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool

MAX_MATCHES = 200
MAX_FILE_BYTES = 1024 * 1024


class GrepInput(BaseModel):
    pattern: str = Field(description="Python regular expression to search for.")
    path: str = Field(default=".", description="File or directory to search.")


def _search(pattern: str, path: str, limit: int) -> tuple[list[str], int]:
    regex = re.compile(pattern)
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    files = [root] if root.is_file() else [p for p in sorted(root.rglob("*")) if p.is_file()]

    hits: list[str] = []
    total = 0
    for file in files:
        try:
            if file.stat().st_size > MAX_FILE_BYTES:
                continue
            text = file.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                total += 1
                if len(hits) < limit:
                    prefix = "" if root.is_file() else f"{file}:"
                    hits.append(f"{prefix}{lineno}:{line}")
    return hits, total


class GrepTool(Tool):
    """Regex search over a file or tree (pure Python; no subprocess).

    Raises FileNotFoundError for a bad path and re.error for an invalid pattern.
    """

    name = "grep"
    description = (
        "Search for a regular expression in a file or recursively in a directory. "
        "Returns matching lines prefixed with their line number."
    )
    InputModel = GrepInput

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, GrepInput)
        hits, total = await asyncio.to_thread(_search, args.pattern, args.path, MAX_MATCHES)
        if not hits:
            return "[no matches]"
        out = "\n".join(hits)
        if total > len(hits):
            out += f"\n[{total - len(hits)} more matches truncated]"
        return out
