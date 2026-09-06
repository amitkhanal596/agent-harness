"""list_dir: list the entries of a directory, one per line."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool

MAX_ENTRIES = 500


class ListDirInput(BaseModel):
    path: str = Field(default=".", description="Directory to list.")


def _entries(path: str, limit: int) -> tuple[list[str], int]:
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    names = sorted(
        f"{p.name}/" if p.is_dir() else p.name for p in directory.iterdir()
    )
    return names[:limit], len(names)


class ListDirTool(Tool):
    """List directory entries. Raises NotADirectoryError / OSError on a bad path."""

    name = "list_dir"
    description = "List the entries of a directory. Directories are suffixed with '/'."
    InputModel = ListDirInput

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, ListDirInput)
        names, total = await asyncio.to_thread(_entries, args.path, MAX_ENTRIES)
        out = "\n".join(names)
        if total > len(names):
            out += f"\n[{total - len(names)} more entries truncated]"
        return out or "[empty directory]"
