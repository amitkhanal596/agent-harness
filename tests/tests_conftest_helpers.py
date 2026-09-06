"""Shared constants for tests that need the raw DSN rather than the pool fixture."""

from __future__ import annotations

import os

DB_URL = os.environ.get("TEST_DATABASE_URL", "postgres://postgres:dev@localhost:5432/agent")
