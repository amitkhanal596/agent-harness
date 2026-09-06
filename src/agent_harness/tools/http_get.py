"""http_get: fetch a URL over HTTP and return the (size-capped) body."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from .base import Tool

MAX_BYTES = 32 * 1024
DEFAULT_TIMEOUT = 20.0


class HttpGetInput(BaseModel):
    url: str = Field(description="Absolute http:// or https:// URL to fetch.")


class HttpGetTool(Tool):
    """GET a URL. Raises ValueError for a non-HTTP scheme and httpx errors on failure."""

    name = "http_get"
    description = (
        f"Fetch an http(s) URL and return the response body, capped at {MAX_BYTES} bytes."
    )
    InputModel = HttpGetInput

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, HttpGetInput)
        if not args.url.startswith(("http://", "https://")):
            raise ValueError(f"unsupported URL scheme: {args.url!r}")
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            resp = await client.get(args.url)
        body = resp.text[:MAX_BYTES]
        suffix = f"\n[truncated at {MAX_BYTES} bytes]" if len(resp.text) > MAX_BYTES else ""
        return f"status={resp.status_code}\n{body}{suffix}"
