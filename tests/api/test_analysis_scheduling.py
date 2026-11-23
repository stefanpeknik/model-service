"""Scheduled analysis runs through the HTTP API: admission, attempts and the execution routes."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from model_service.api.app_factory import analysis_resolver, create_app
from model_service.api.scheduler.dispatch import RemoteFailureError
from model_service.api.scheduler.execution_scheduler import ExecutionScheduler
from model_service.api.scheduler.internal_state import RetryPolicy
from model_service.catalog import Catalog, load_catalog
from model_service.catalog.analyses import resolve_run
from model_service.contracts.result import RunResult, SharedArtifact
from model_service.formats import score_grid
from model_service.providers.standalone.adapter import StandaloneProviderAdapter
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from tests.support.memory_repository import InMemoryExecutionRepository


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from model_service.contracts.analysis_http import AnalysisRunRequest


pytestmark = pytest.mark.anyio

CATALOG = load_catalog(Path(__file__).resolve().parents[2] / "catalog", "rationai-test")
SCHEDULE = "/v1/analyses/prostate-heatmap/schedule"
SETTINGS = {"min_tissue_fraction": 0.0, "skip_background": True, "stride_fraction": 1.0}


class FakeRunner:
    """The analysis runner: records each attempt, failing the first ones if told to."""

    def __init__(self, root: Path, failures: int = 0) -> None:
        self.requests: list[AnalysisRunRequest] = []
        self.root = root
        self.failures = failures

    async def run_analysis(self, request: AnalysisRunRequest) -> RunResult:
        self.requests.append(request)
        if len(self.requests) <= self.failures:
            raise RemoteFailureError(
                code="model_failed", safe_message="try again", retryable=True
            )
        return RunResult(
            {
                "heatmap": SharedArtifact(
                    str(self.root / "out.npz"), score_grid.MEDIA_TYPE
                )
            }
        )


class Deployment:
    """The API with a real scheduler over the in-memory repository and a fake runner."""

    def __init__(self, root: Path, runner: FakeRunner) -> None:
        self.catalog = CATALOG
        self.runner = runner
        self.scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(
                processing_policy=RetryPolicy(3, timedelta(0), timedelta(0))
            ),
            dispatch=runner,
            providers={
                "standalone": StandaloneProviderAdapter(
                    SharedStorageAccessFactory([root])
                )
            },
            resolve_analysis=analysis_resolver(lambda: self.catalog),
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            poll_interval=0.01,
        )

    def replace_catalog(self, catalog: Catalog) -> None:
        self.catalog = catalog

    async def client(self) -> AsyncIterator[AsyncClient]:
        app = create_app(self.scheduler, catalog=lambda: self.catalog)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://model-service"
        ) as client:
            yield client


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "slide.mrxs").touch()
    return tmp_path


def body(root: Path, **changes: object) -> dict[str, object]:
    return {
        "inputs": {"slide": {"kind": "shared_path", "path": "slide.mrxs"}},
        "input_access": {"kind": "shared_storage", "state": {"root": str(root)}},
        "configuration": {"stride_fraction": 1.0},
        "output_path": "out.npz",
        **changes,
    }


async def wait_for(client: AsyncClient, execution_id: str) -> dict[str, object]:
    async with asyncio.timeout(5):
        while True:
            view = (await client.get(f"/v1/executions/{execution_id}")).json()
            if view["status"] in {"succeeded", "failed"}:
                return view
            await asyncio.sleep(0.01)


async def test_admission_records_the_resolved_analysis(root: Path) -> None:
    deployment = Deployment(root, FakeRunner(root))
    headers = {"Idempotency-Key": "run-1"}
    async for client in deployment.client():
        created = await client.post(SCHEDULE, json=body(root), headers=headers)
        replay = await client.post(SCHEDULE, json=body(root), headers=headers)
        execution_id = created.json()["execution"]["id"]
        polled = await client.get(f"/v1/executions/{execution_id}")
        looked_up = await client.post(
            "/v1/executions/lookup", json={"ids": [execution_id]}
        )
        cancelled = await client.post(f"/v1/executions/{execution_id}/cancel")

    expected = resolve_run(CATALOG, "prostate-heatmap", {"stride_fraction": 1.0})
    assert created.status_code == 202, created.text
    assert (replay.status_code, replay.json()["execution"]["id"]) == (
        200,
        execution_id,
    )
    assert polled.json() == {
        "id": execution_id,
        "analysis_id": "prostate-heatmap",
        "models": ["prostate-classifier-1"],
        "analysis_digest": expected.digest,
        "status": "queued",
        "configuration": SETTINGS,
        "result": None,
        "error": None,
        "provider_pending": False,
    }
    assert looked_up.json() == [polled.json()]
    assert cancelled.json()["status"] == "cancelled"


async def test_every_attempt_sends_the_document_admitted(root: Path) -> None:
    deployment = Deployment(root, FakeRunner(root, failures=1))
    async for client in deployment.client():
        created = await client.post(
            SCHEDULE, json=body(root), headers={"Idempotency-Key": "run-1"}
        )
        # The analysis leaves the catalog after admission; the run doesn't care.
        deployment.replace_catalog(Catalog(None))
        async with deployment.scheduler.running():
            view = await wait_for(client, created.json()["execution"]["id"])

    assert view["status"] == "succeeded", view
    expected = resolve_run(CATALOG, "prostate-heatmap", {"stride_fraction": 1.0})
    first, retry = deployment.runner.requests
    assert first.analysis == retry.analysis == expected.document
    assert retry.invocation.configuration == SETTINGS
    assert first.execution_context.processing_attempt_id != (
        retry.execution_context.processing_attempt_id
    )


@pytest.mark.parametrize(
    ("path", "changes", "status", "code"),
    [
        ("/v1/analyses/missing/schedule", {}, 404, "unknown_analysis"),
        (SCHEDULE, {"configuration": {"stride_fraction": -1}}, 422, "invalid_request"),
        (SCHEDULE, {"inputs": {}}, 422, "invalid_request"),
        (
            SCHEDULE,
            {"inputs": {"slide": {"kind": "shared_path", "path": "absent.mrxs"}}},
            422,
            "invalid_request",
        ),
        (
            SCHEDULE,
            {"input_access": {"kind": "shared_storage", "state": {"root": "/"}}},
            422,
            "invalid_request",
        ),
    ],
)
async def test_a_run_the_analysis_cannot_take_is_not_admitted(
    root: Path, path: str, changes: dict[str, object], status: int, code: str
) -> None:
    deployment = Deployment(root, FakeRunner(root))
    async for client in deployment.client():
        refused = await client.post(
            path, json=body(root, **changes), headers={"Idempotency-Key": "run-1"}
        )
        # Nothing was admitted under the key: a valid request creates.
        corrected = await client.post(
            SCHEDULE, json=body(root), headers={"Idempotency-Key": "run-1"}
        )

    assert (refused.status_code, refused.json()["detail"]["code"]) == (status, code)
    assert corrected.status_code == 202, corrected.text
