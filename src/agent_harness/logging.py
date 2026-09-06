"""Structured JSON logging with run/step correlation.

Trace data lives in Postgres, which is the right home for it but useless when the
database itself is the thing that is broken, and useless for tailing a run in flight.
These logs are the complementary channel: one JSON object per line on stdout, every
line carrying `run_id` and `step` so an operator can filter a single run out of an
aggregator.

Correlation is carried in a `ContextVar`, so it survives across `await` boundaries and
concurrent tasks without being threaded through every function signature.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

#: Correlation fields applied to every record emitted inside a `log_context` block.
_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)

#: Attributes the stdlib puts on every LogRecord; anything else is caller-supplied and
#: gets merged into the JSON payload.
_STANDARD_FIELDS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
        "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
        "message", "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as one JSON object, merging correlation and extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_CONTEXT.get() or {})
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Install the root handler. Idempotent — repeated calls replace the handler.

    Logs go to stderr so stdout stays clean for CLI output that may be piped.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if json_format else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach `fields` to every log record emitted inside this block.

    Nests: inner blocks add to the outer context rather than replacing it, so a step
    context inherits the run context.
    """
    token = _CONTEXT.set({**(_CONTEXT.get() or {}), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def get_logger(name: str) -> logging.Logger:
    """A logger for `name`, conventionally the module `__name__`."""
    return logging.getLogger(name)
