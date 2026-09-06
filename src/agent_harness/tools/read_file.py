"""read_file: return a file's contents, truncated to a size cap."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from .base import Tool

MAX_BYTES = 64 * 1024


class ReadFileInput(BaseModel):
    path: str = Field(description="Path to the file to read.")


def _read(path: str, limit: int) -> tuple[str, bool]:
    """Read at most `limit` bytes, reporting whether more remained.

    Reads only `limit + 1` bytes rather than slurping the whole file and slicing: a
    model can point this at a multi-gigabyte log, and loading it into memory just to
    discard all but 64 KB is an easy OOM. The extra byte is what distinguishes
    "exactly at the cap" from "there is more".
    """
    with open(path, "rb") as handle:
        data = handle.read(limit + 1)
    truncated = len(data) > limit
    return data[:limit].decode("utf-8", errors="replace"), truncated


class ReadFileTool(Tool):
    """Read a UTF-8 text file. Raises OSError if the path is missing or unreadable."""

    name = "read_file"
    description = (
        f"Read the contents of a text file. Output is truncated at {MAX_BYTES} bytes."
    )
    InputModel = ReadFileInput

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, ReadFileInput)
        text, truncated = await asyncio.to_thread(_read, args.path, MAX_BYTES)
        if truncated:
            text += f"\n\n[truncated at {MAX_BYTES} bytes]"
        return text
