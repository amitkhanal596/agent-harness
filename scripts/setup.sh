#!/usr/bin/env bash
# Executable version of README §12. Idempotent: safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
DSN="${DATABASE_URL:-postgres://postgres:dev@localhost:5432/agent}"

echo "==> creating .venv"
[ -d .venv ] || "$PY" -m venv .venv

echo "==> installing the package (editable, with dev extras)"
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -e ".[dev]"

echo "==> checking Postgres at $DSN"
if ./.venv/bin/python - "$DSN" <<'PYEOF'
import asyncio, sys
import asyncpg

async def main() -> int:
    try:
        conn = await asyncpg.connect(sys.argv[1], timeout=3)
    except Exception as exc:
        print(f"    not reachable: {exc}")
        return 1
    await conn.close()
    return 0

sys.exit(asyncio.run(main()))
PYEOF
then
    echo "    reachable"
else
    cat <<'MSG'
    Postgres is not reachable. Start one with either:

      docker run --name agent-pg -e POSTGRES_PASSWORD=dev -p 5432:5432 -d postgres:15

    or, with a local (Homebrew) server:

      createdb agent
      psql -d agent -c "CREATE ROLE postgres LOGIN SUPERUSER PASSWORD 'dev';"

    then re-run this script.
MSG
    exit 1
fi

echo "==> writing .env if missing"
[ -f .env ] || { cp .env.example .env; echo "    created .env — fill in ANTHROPIC_API_KEY"; }

echo "==> applying migrations"
./.venv/bin/agent migrate

cat <<'MSG'

Setup complete. Next:

  source .venv/bin/activate
  pytest && ruff check . && mypy src     # verify the build
  agent eval --mock                      # run the eval set offline
  agent run "read ./config.txt and tell me the configured port"   # needs an API key
MSG
