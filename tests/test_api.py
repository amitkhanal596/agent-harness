"""§2.6: the HTTP service — one shared pool, same engine as the CLI."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_harness.api import create_app
from agent_harness.config import Settings
from agent_harness.models import Completion, ToolCall
from helpers import MockLLMClient


def _settings(db_url: str) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=db_url,
        max_steps=5,
        tool_timeout_seconds=10.0,
        tool_max_retries=0,
        run_budget_usd=1.0,
        log_json=False,
    )


def _script(temp_repo: Path) -> list[Completion]:
    return [
        Completion(
            stop_reason="tool_use",
            prompt_tokens=100,
            output_tokens=20,
            tool_calls=[
                ToolCall(id="t1", name="read_file",
                         input={"path": str(temp_repo / "config.txt")})
            ],
        ),
        Completion(stop_reason="end_turn", text="the port is 8080",
                   prompt_tokens=150, output_tokens=25),
    ]


@pytest_asyncio.fixture
async def client(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> AsyncIterator[AsyncClient]:
    app = create_app(
        settings=_settings("unused-because-pool-is-injected"),
        llm=MockLLMClient(_script(temp_repo), repeat_last=False),
        pool=db_pool,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        async with app.router.lifespan_context(app):
            yield http


async def test_health_checks_the_database(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_create_run_executes_end_to_end(client: AsyncClient) -> None:
    resp = await client.post("/runs", json={"task": "read the config"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "completed"
    assert body["output"] == "the port is 8080"
    assert body["steps"] == 1
    assert body["run_id"]


async def test_get_run_returns_persisted_summary(client: AsyncClient) -> None:
    run_id = (await client.post("/runs", json={"task": "read the config"})).json()["run_id"]
    resp = await client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["task"] == "read the config"
    assert body["total_tokens"] == 295  # 100+20 then 150+25
    assert body["total_cost_usd"] > 0
    assert body["final_output"] == "the port is 8080"


async def test_trace_endpoint_matches_the_events(client: AsyncClient) -> None:
    run_id = (await client.post("/runs", json={"task": "read the config"})).json()["run_id"]
    resp = await client.get(f"/runs/{run_id}/trace")
    assert resp.status_code == 200
    body = resp.json()
    kinds = [e["kind"] for e in body["events"]]
    assert kinds.count("llm_call") == 2
    assert kinds.count("tool_call") == 1
    assert sum(
        (e["prompt_tokens"] or 0) + (e["output_tokens"] or 0) for e in body["events"]
    ) == body["total_tokens"]


async def test_checkpoints_endpoint_is_replay_over_http(client: AsyncClient) -> None:
    run_id = (await client.post("/runs", json={"task": "read the config"})).json()["run_id"]
    resp = await client.get(f"/runs/{run_id}/checkpoints")
    assert resp.status_code == 200
    steps = [c["step"] for c in resp.json()["checkpoints"]]
    assert steps == sorted(steps)
    assert resp.json()["checkpoints"][-1]["status"] == "completed"


async def test_unknown_run_is_404(client: AsyncClient) -> None:
    from uuid import uuid4

    missing = uuid4()
    assert (await client.get(f"/runs/{missing}")).status_code == 404
    assert (await client.get(f"/runs/{missing}/trace")).status_code == 404
    assert (await client.get(f"/runs/{missing}/checkpoints")).status_code == 404


async def test_malformed_uuid_is_rejected(client: AsyncClient) -> None:
    assert (await client.get("/runs/not-a-uuid")).status_code == 422


async def test_empty_task_is_rejected(client: AsyncClient) -> None:
    assert (await client.post("/runs", json={"task": ""})).status_code == 422


async def test_consistency_endpoint_reports_health(client: AsyncClient) -> None:
    resp = await client.get("/health/consistency")
    assert resp.status_code == 200
    assert "orphaned_events" in resp.json()
    assert "stale_runs" in resp.json()


async def test_budget_exceeded_returns_a_failed_run_not_a_500(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> None:
    """A guardrail trip is a reported outcome, not a server error."""
    settings = _settings("unused")
    settings.run_budget_usd = 0.0000001
    app = create_app(
        settings=settings,
        llm=MockLLMClient(
            [Completion(stop_reason="end_turn", text="x", prompt_tokens=100_000,
                        output_tokens=0)]
        ),
        pool=db_pool,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        async with app.router.lifespan_context(app):
            resp = await http.post("/runs", json={"task": "expensive"})
    assert resp.status_code == 201
    assert resp.json()["status"] == "failed"
    assert "budget" in resp.json()["error"]


async def test_concurrent_requests_share_one_pool(
    db_pool: asyncpg.Pool, temp_repo: Path
) -> None:
    """The §2.6 point: many concurrent runs on a single process-lifetime pool."""
    app = create_app(
        settings=_settings("unused"),
        llm=MockLLMClient(
            [Completion(stop_reason="end_turn", text="ok", prompt_tokens=10,
                        output_tokens=5)],
            repeat_last=True,
        ),
        pool=db_pool,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        async with app.router.lifespan_context(app):
            responses = await asyncio.gather(
                *(http.post("/runs", json={"task": f"task {i}"}) for i in range(12))
            )

    assert all(r.status_code == 201 for r in responses)
    assert all(r.json()["status"] == "completed" for r in responses)
    # Every run got a distinct id — no state bleed between concurrent requests.
    assert len({r.json()["run_id"] for r in responses}) == 12


async def test_openapi_schema_is_generated(client: AsyncClient) -> None:
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/runs" in paths
    assert "/runs/{run_id}/trace" in paths
