"""Check the public API: scheduling, executions, disconnects, health and safe errors."""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from httpx import (
    ASGITransport,
    AsyncClient,
    ConnectError,
    MockTransport,
    Request,
    Response,
)

from model_service.api.app_factory import analysis_resolver, create_app
from model_service.api.health import DeploymentHealth
from model_service.api.scheduler.execution_scheduler import ExecutionScheduler
from model_service.api.scheduler.public import (
    AnalysisScheduleRequest,
    ExecutionView,
    ScheduleResult,
    SchedulingUnavailableError,
)
from model_service.api.scheduler.status import ExecutionStatus
from model_service.api.scheduler.store.repository import (
    ExecutionRepositoryUnavailableError,
)
from model_service.api.synchronous_runs import SynchronousAnalysisService
from model_service.catalog import load_catalog
from model_service.catalog.analyses import resolve_run
from model_service.contracts.result import BuilderResult, InlineOutput
from model_service.providers.standalone.adapter import StandaloneProviderAdapter
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from tests.support.memory_repository import InMemoryExecutionRepository


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from starlette.types import Message, Scope

    from model_service.api.scheduler.internal_state import ExecutionRecord
    from model_service.catalog import Catalog
    from model_service.contracts.analysis_http import AnalysisRunRequest

    type Handler = Callable[[AnalysisRunRequest], Awaitable[BuilderResult]]


CATALOG = load_catalog(Path(__file__).resolve().parents[2] / "catalog", "rationai-test")
RUN = "/v1/analyses/prostate-heatmap/run"
SCHEDULE = "/v1/analyses/prostate-heatmap/schedule"


def catalog() -> Catalog:
    return CATALOG


# Every prostate-heatmap setting at its default: what a run without
# configuration records.
HEATMAP_DEFAULTS = {
    "stride_fraction": 0.5,
    "skip_background": True,
    "min_tissue_fraction": 0.0,
}


class FakeDispatch:
    """Record dispatched runs and delegate to a test handler."""

    def __init__(self, handler: Handler | None = None) -> None:
        self.calls: list[AnalysisRunRequest] = []
        self._handler = handler

    async def run_analysis(self, request: AnalysisRunRequest) -> BuilderResult:
        self.calls.append(request)
        if self._handler is None:
            return BuilderResult(outputs={"score": InlineOutput(value=0.9)})
        return await self._handler(request)


class RecordingScheduler:
    """Record admitted requests and return a fixed queued execution."""

    alive = True

    def __init__(self) -> None:
        self.requests: list[AnalysisScheduleRequest] = []

    async def check_ready(self) -> None:
        return None

    async def schedule(self, request: AnalysisScheduleRequest) -> ScheduleResult:
        self.requests.append(request)
        return ScheduleResult(
            execution=ExecutionView(
                id="execution-1",
                analysis_id=request.analysis_id,
                models=[],
                analysis_digest=None,
                status=ExecutionStatus.QUEUED,
            ),
            created=True,
        )

    async def get(self, execution_id: str) -> ExecutionView:
        raise AssertionError(execution_id)

    async def get_many(self, execution_ids: list[str]) -> list[ExecutionView]:
        raise AssertionError(execution_ids)

    async def cancel(self, execution_id: str) -> ExecutionView:
        raise AssertionError(execution_id)


class ApiTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "slide.svs").touch()
        (self.root / "roi.tiff").touch()
        self.dispatch = FakeDispatch()
        self.repository = InMemoryExecutionRepository()
        self.scheduler = self.make_scheduler(self.repository, self.dispatch)

    def make_scheduler(
        self, repository: InMemoryExecutionRepository, dispatch: FakeDispatch
    ) -> ExecutionScheduler:
        return ExecutionScheduler(
            repository=repository,
            dispatch=dispatch,
            providers={
                "standalone": StandaloneProviderAdapter(
                    SharedStorageAccessFactory([self.root])
                )
            },
            resolve_analysis=analysis_resolver(catalog),
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            poll_interval=0.01,
        )

    def client(self, app: FastAPI, *, raise_app_exceptions: bool = True) -> AsyncClient:
        return AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions),
            base_url="http://model-service",
        )

    def app(self, **kwargs: object) -> FastAPI:
        options: dict[str, object] = {
            "catalog": catalog,
            "analysis_runner": SynchronousAnalysisService(catalog, self.dispatch),
        }
        options.update(kwargs)
        return create_app(self.scheduler, **options)  # type: ignore[arg-type]

    def request(self) -> dict[str, object]:
        return {
            "inputs": {
                "slide": {"kind": "shared_path", "path": "slide.svs"},
                "region_of_interest": {"kind": "shared_path", "path": "roi.tiff"},
            },
            "input_access": {
                "kind": "shared_storage",
                "state": {"root": str(self.root)},
            },
        }

    def run_request(self) -> dict[str, object]:
        return {
            **self.request(),
            "output_path": "outputs/heatmap.npz",
            "deadline_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        }

    def schedule_request(self) -> dict[str, object]:
        return self.request()

    async def schedule(
        self,
        client: AsyncClient,
        body: dict[str, object] | None = None,
        *,
        key: str | None = "request-1",
        url: str = SCHEDULE,
    ) -> Response:
        return await client.post(
            url,
            json=self.schedule_request() if body is None else body,
            headers={} if key is None else {"Idempotency-Key": key},
        )

    def assert_error(
        self,
        response: Response,
        status_code: int,
        code: str,
        *,
        retryable: bool = False,
    ) -> None:
        self.assertEqual(response.status_code, status_code, response.text)
        self.assertEqual(response.json()["detail"]["code"], code)
        self.assertEqual(response.json()["detail"]["retryable"], retryable)


class ScheduleRouteTests(ApiTestCase):
    async def test_schedule_replay_poll_and_cancel(self) -> None:
        async with self.client(self.app()) as client:
            created = await self.schedule(client)
            replay = await self.schedule(client)
            execution_id = created.json()["execution"]["id"]
            polled = await client.get(f"/v1/executions/{execution_id}")
            cancelled = await client.post(f"/v1/executions/{execution_id}/cancel")
            unknown = await client.get("/v1/executions/missing")

        self.assertEqual(created.status_code, 202, created.text)
        self.assertTrue(created.json()["created"])
        self.assertEqual(replay.status_code, 200)
        self.assertFalse(replay.json()["created"])
        self.assertEqual(replay.json()["execution"]["id"], execution_id)
        self.assertEqual(
            polled.json(),
            {
                "id": execution_id,
                "analysis_id": "prostate-heatmap",
                "models": ["prostate-classifier-1"],
                "analysis_digest": resolve_run(
                    CATALOG, "prostate-heatmap", None
                ).digest,
                "status": "queued",
                "configuration": HEATMAP_DEFAULTS,
                "result": None,
                "error": None,
                "provider_pending": False,
            },
        )
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assert_error(unknown, 404, "unknown_execution")

    async def test_accepted_settings_are_the_ones_the_execution_runs_with(
        self,
    ) -> None:
        body = {**self.schedule_request(), "configuration": {"stride_fraction": 1.0}}
        async with self.client(self.app()) as client:
            created = await self.schedule(client, body)
            execution = await client.get(
                f"/v1/executions/{created.json()['execution']['id']}"
            )

        self.assertEqual(created.status_code, 202, created.text)
        self.assertEqual(
            execution.json()["configuration"],
            HEATMAP_DEFAULTS | {"stride_fraction": 1.0},
        )

    async def test_a_setting_the_analysis_rejects_is_refused_naming_it(self) -> None:
        body = {**self.schedule_request(), "configuration": {"stride_fraction": 0}}
        async with self.client(self.app()) as client:
            refused = await self.schedule(client, body)
            # Nothing was admitted under the key: a valid request creates.
            corrected = await self.schedule(client)

        self.assert_error(refused, 422, "invalid_request")
        self.assertIn("stride_fraction", refused.json()["detail"]["message"])
        self.assertTrue(corrected.json()["created"])

    async def test_a_replay_that_changes_the_request_is_refused(self) -> None:
        changed = {**self.schedule_request(), "configuration": {"stride_fraction": 1.0}}
        async with self.client(self.app()) as client:
            created = await self.schedule(client)
            replay = await self.schedule(client, changed)
            execution = await client.get(
                f"/v1/executions/{created.json()['execution']['id']}"
            )

        self.assert_error(replay, 409, "schedule_conflict")
        self.assertEqual(execution.json()["configuration"], HEATMAP_DEFAULTS)

    async def test_lookup_and_validation_errors_are_distinct(self) -> None:
        body = self.schedule_request()
        async with self.client(self.app()) as client:
            unknown_analysis = await self.schedule(
                client, url="/v1/analyses/missing/schedule"
            )
            empty_key = await self.schedule(client, key="")
            no_key = await self.schedule(client, key=None)
            key_in_body = await self.schedule(
                client, {**body, "idempotency_key": "request-1"}, key=None
            )
            bad_timeout = await self.schedule(
                client, {**body, "requested_timeout_seconds": 0}
            )
            provider = await self.schedule(client, {**body, "provider": "standalone"})

        self.assert_error(unknown_analysis, 404, "unknown_analysis")
        for response in (
            empty_key,
            no_key,
            key_in_body,
            bad_timeout,
            provider,
        ):
            self.assert_error(response, 422, "invalid_request")

    async def test_admission_checks_inputs_and_allowed_roots(self) -> None:
        request = self.schedule_request()
        with TemporaryDirectory() as other:
            cases = [
                {**request, "inputs": {}},
                {
                    **request,
                    "inputs": {"slide": {"kind": "shared_path", "path": "absent.svs"}},
                },
                {
                    **request,
                    "input_access": {
                        "kind": "shared_storage",
                        "state": {"root": other},
                    },
                },
            ]
            async with self.client(self.app()) as client:
                responses = [
                    await self.schedule(client, case, key=f"case-{i}")
                    for i, case in enumerate(cases)
                ]

        for response in responses:
            self.assert_error(response, 422, "invalid_request")
        self.assertIn("not allowed", responses[2].json()["detail"]["message"])


class DisconnectTests(ApiTestCase):
    async def test_client_disconnect_cancels_dispatch(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def wait_forever(*_args: object) -> BuilderResult:
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("dispatch unexpectedly completed")
            finally:
                cancelled.set()

        self.dispatch = FakeDispatch(wait_forever)
        app = self.app()
        body = json.dumps(self.run_request()).encode()
        sent = False

        async def receive() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await started.wait()
            return {"type": "http.disconnect"}

        async def send(_message: Message) -> None:
            return None

        path = RUN
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"model-service"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("model-service", 80),
            "state": {},
        }

        with self.assertRaises(asyncio.CancelledError):
            await app(scope, receive, send)
        await asyncio.wait_for(cancelled.wait(), timeout=1)


class HealthAndAvailabilityTests(ApiTestCase):
    def health(self, client: AsyncClient) -> DeploymentHealth:
        return DeploymentHealth(
            client,
            origin="http://ray",
            routes=lambda: ["/analyses", "/prostate-classifier-1"],
        )

    async def test_an_api_without_a_running_worker_is_not_live(self) -> None:
        async with self.client(self.app()) as client:
            stopped = await client.get("/healthz")
            async with self.scheduler.running():
                live = await client.get("/healthz")

        self.assertEqual(stopped.status_code, 503)
        self.assertEqual(live.status_code, 200)

    async def test_missing_routes_are_separate_from_readiness(self) -> None:
        routes = {"/analyses/": "analysis-runner"}
        async with (
            AsyncClient(
                transport=MockTransport(lambda _request: Response(200, json=routes))
            ) as ray,
            self.client(self.app(deployment_health=self.health(ray))) as client,
            self.scheduler.running(),
        ):
            live = await client.get("/healthz")
            ready = await client.get("/readyz")
            deployments = await client.get("/healthz/deployments")

        self.assertEqual(live.status_code, 200)
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(deployments.status_code, 503)
        self.assertEqual(
            deployments.json(),
            {
                "status": "missing_applications",
                "missing_routes": ["/prostate-classifier-1"],
            },
        )

    async def test_all_routes_present_is_ready(self) -> None:
        routes = {"/analyses": "analysis-runner", "/prostate-classifier-1": "p"}
        async with (
            AsyncClient(
                transport=MockTransport(lambda _request: Response(200, json=routes))
            ) as ray,
            self.client(self.app(deployment_health=self.health(ray))) as client,
        ):
            deployments = await client.get("/healthz/deployments")
        async with self.client(self.app()) as client:
            unconfigured = await client.get("/healthz/deployments")

        self.assertEqual(deployments.status_code, 200)
        self.assertEqual(deployments.json()["status"], "ready")
        self.assertEqual(unconfigured.json()["status"], "not_configured")

    async def test_ray_outage_keeps_durable_api_usable(self) -> None:
        def unavailable(request: Request) -> Response:
            raise ConnectError("Ray is unavailable", request=request)

        async with (
            AsyncClient(transport=MockTransport(unavailable)) as ray,
            self.client(self.app(deployment_health=self.health(ray))) as client,
        ):
            ready = await client.get("/readyz")
            deployments = await client.get("/healthz/deployments")
            admitted = await self.schedule(client)
            execution_id = admitted.json()["execution"]["id"]
            polled = await client.get(f"/v1/executions/{execution_id}")
            cancelled = await client.post(f"/v1/executions/{execution_id}/cancel")

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(deployments.status_code, 503)
        self.assertEqual(deployments.json()["status"], "unreachable")
        self.assertEqual(admitted.status_code, 202)
        self.assertEqual(polled.json()["status"], "queued")
        self.assertEqual(cancelled.json()["status"], "cancelled")

    async def test_database_outage_is_retryable(self) -> None:
        class UnavailableRepository(InMemoryExecutionRepository):
            async def check_ready(self) -> None:
                raise ExecutionRepositoryUnavailableError("database unavailable")

            async def get(self, execution_id: str) -> ExecutionRecord:
                raise ExecutionRepositoryUnavailableError(execution_id)

            async def request_cancel(self, execution_id: str) -> ExecutionRecord:
                raise ExecutionRepositoryUnavailableError(execution_id)

            async def find_submission(self, identity: object) -> None:
                raise ExecutionRepositoryUnavailableError("database unavailable")

        self.scheduler = self.make_scheduler(UnavailableRepository(), self.dispatch)
        async with (
            self.client(self.app(), raise_app_exceptions=False) as client,
            self.scheduler.running(),
        ):
            responses = [
                await client.get("/readyz"),
                await client.get("/v1/executions/execution-1"),
                await client.post("/v1/executions/execution-1/cancel"),
                await self.schedule(client),
            ]
            live = await client.get("/healthz")

        for response in responses:
            self.assert_error(response, 503, "scheduling_unavailable", retryable=True)
        self.assertEqual(live.status_code, 200)

    async def test_scheduler_unavailability_is_retryable(self) -> None:
        class UnavailableScheduler(RecordingScheduler):
            async def schedule(
                self, request: AnalysisScheduleRequest
            ) -> ScheduleResult:
                raise SchedulingUnavailableError("temporarily unavailable")

        app = create_app(UnavailableScheduler(), catalog=catalog)
        async with self.client(app) as client:
            response = await self.schedule(client)

        self.assert_error(response, 503, "scheduling_unavailable", retryable=True)


if __name__ == "__main__":
    unittest.main()
