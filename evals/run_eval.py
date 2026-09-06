"""Standalone entry point for the eval set: `python evals/run_eval.py [tasks.jsonl]`.

Equivalent to `agent eval`; both call `agent_harness.evals_runner.run_eval_set`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agent_harness.evals_runner import run_eval_set

DEFAULT_TASKS = Path(__file__).with_name("tasks.jsonl")


def main() -> None:
    tasks = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_TASKS)
    report = asyncio.run(run_eval_set(tasks))
    print(report["text"])
    sys.exit(0 if report["pass_rate"] == 1.0 else 1)


if __name__ == "__main__":
    main()
