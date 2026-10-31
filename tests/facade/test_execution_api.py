"""Check the public API: catalog listings, builder runs, scheduling and safe errors."""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import lz4.frame
from httpx import (
    ASGITransport,
    AsyncClient,
    ConnectError,
    MockTransport,
    Request,
    Response,
)

from model_service.builders.client import (
    ConfirmedRejectionError,
    RemoteFailureError,
    UnknownOutcomeError,
)
from model_service.builders.contracts import (
    BuilderResult,
    InlineOutput,
    SharedArtifact,
    SpatialScoresModel,
    TileEmbeddingModel,
)
from model_service.facade.api import create_app
from model_service.facade.app import model_resolver
from model_service.facade.catalog import ModelCatalog
from model_service.facade.health import DeploymentHealth
from model_service.facade.inference import HttpInferenceProxy, ModelInference
from model_service.facade.synchronous import SynchronousBuilderService
from model_service.providers.shared_storage import SharedStorageAccessAdapter
from model_service.providers.standalone import StandaloneProviderAdapter
from model_service.scheduler.contracts import (
    ExecutionStatus,
    ExecutionView,
    Operation,
    ScheduleRequest,
    ScheduleResult,
    SchedulingUnavailableError,
)
from model_service.scheduler.scheduler import ExecutionScheduler
from model_service.scheduler.store.repository import ExecutionRepositoryUnavailableError
from tests.support.memory_repository import InMemoryExecutionRepository


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from starlette.types import Message, Scope

    from model_service.builders.contracts import (
        ExecutionContext,
        Invocation,
        ModelBinding,
    )
    from model_service.scheduler.domain import ExecutionRecord

    type Handler = Callable[
        [str, Invocation, ModelBinding, ExecutionContext], Awaitable[BuilderResult]
    ]


PROSTATE = SpatialScoresModel(
    id="prostate-classifier-1",
    tile_size=512,
    output_tile_size=1,
    n_channels=1,
    mpp=0.468,
)
VIRCHOW = TileEmbeddingModel(
    id="virchow2", tile_size=224, mpp=0.5, summaries=("global", "local_mean")
)
CATALOG = ModelCatalog([PROSTATE, VIRCHOW])
RUN = "/v1/builders/heatmap/models/prostate-classifier-1/run"
SCHEDULE = "/v1/builders/heatmap/models/prostate-classifier-1/schedule"


class FakeDispatch:
    """Record dispatched calls and delegate to a test handler."""

    def __init__(self, handler: Handler | None = None) -> None:
        self.calls: list[tuple[str, Invocation, ModelBinding, ExecutionContext]] = []
        self._handler = handler

    async def run(
        self,
        builder_id: str,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext,
    ) -> BuilderResult:
        self.calls.append((builder_id, invocation, model, execution_context))
        if self._handler is None:
            return BuilderResult(outputs={"score": InlineOutput(value=0.9)})
        return await self._handler(builder_id, invocation, model, execution_context)


class RecordingScheduler:
    """Record admitted requests and return a fixed queued execution."""

    def __init__(self) -> None:
        self.requests: list[ScheduleRequest] = []

    async def check_ready(self) -> None:
        return None

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        self.requests.append(request)
        return ScheduleResult(
            execution=ExecutionView(
                id="execution-1",
                operation=Operation.BUILDER,
                builder_id=request.builder_id,
                model_id=request.model_id,
                status=ExecutionStatus.QUEUED,
            ),
            created=True,
        )

    async def get(self, execution_id: str) -> ExecutionView:
        raise AssertionError(execution_id)

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
                    SharedStorageAccessAdapter([self.root])
                )
            },
            resolve_model=model_resolver(CATALOG),
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
            "catalog": CATALOG,
            "builder_runner": SynchronousBuilderService(CATALOG, self.dispatch),
        }
        options.update(kwargs)
        return create_app(self.scheduler, **options)  # type: ignore[arg-type]

    def builder_request(self) -> dict[str, object]:
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
            **self.builder_request(),
            "output_path": "outputs/heatmap.tiff",
            "deadline_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        }

    def schedule_request(self) -> dict[str, object]:
        return self.builder_request()

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


class CatalogRouteTests(ApiTestCase):
    async def test_models_list_contracts_parameters_and_compatible_builders(
        self,
    ) -> None:
        async with self.client(self.app()) as client:
            listed = await client.get("/v1/models")
            single = await client.get("/v1/models/virchow2")
            missing = await client.get("/v1/models/missing")

        self.assertEqual(
            listed.json(),
            [
                {
                    "id": "prostate-classifier-1",
                    "contract": "spatial-scores",
                    "parameters": {
                        "tile_size": 512,
                        "mpp": 0.468,
                        "output_tile_size": 1,
                        "n_channels": 1,
                    },
                    "builders": ["heatmap"],
                },
                {
                    "id": "virchow2",
                    "contract": "tile-embedding",
                    "parameters": {
                        "tile_size": 224,
                        "mpp": 0.5,
                        "summaries": ["global", "local_mean"],
                    },
                    "builders": ["tile-embeddings"],
                },
            ],
        )
        self.assertEqual(single.json()["id"], "virchow2")
        self.assert_error(missing, 404, "unknown_model")

    async def test_builders_list_contracts_io_and_default_configuration(self) -> None:
        async with self.client(self.app()) as client:
            response = await client.get("/v1/builders")

        builders = {item["id"]: item for item in response.json()}
        self.assertEqual(set(builders), {"heatmap", "tile-embeddings"})
        heatmap = builders["heatmap"]
        self.assertEqual(heatmap["accepts"], "spatial-scores")
        self.assertEqual(
            heatmap["inputs"]["region_of_interest"],
            {"required": False, "media_type": "image/tiff"},
        )
        self.assertTrue(heatmap["inputs"]["slide"]["required"])
        self.assertEqual(
            heatmap["outputs"],
            {"heatmap": {"kind": "shared_artifact", "media_type": "application/x-npz"}},
        )
        self.assertEqual(heatmap["configuration"]["stride_fraction"], 0.5)
        self.assertEqual(builders["tile-embeddings"]["accepts"], "tile-embedding")


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
                "operation": "builder",
                "builder_id": "heatmap",
                "model_id": "prostate-classifier-1",
                "status": "queued",
                "result": None,
                "error": None,
                "provider_pending": False,
                "message": None,
            },
        )
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assert_error(unknown, 404, "unknown_execution")

    async def test_lookup_and_validation_errors_are_distinct(self) -> None:
        body = self.schedule_request()
        async with self.client(self.app()) as client:
            unknown_builder = await self.schedule(
                client, url="/v1/builders/missing/models/prostate-classifier-1/schedule"
            )
            unknown_model = await self.schedule(
                client, url="/v1/builders/heatmap/models/missing/schedule"
            )
            incompatible = await self.schedule(
                client, url="/v1/builders/heatmap/models/virchow2/schedule"
            )
            empty_key = await self.schedule(client, key="")
            no_key = await self.schedule(client, key=None)
            key_in_body = await self.schedule(
                client, {**body, "idempotency_key": "request-1"}, key=None
            )
            settings = await self.schedule(
                client, {**body, "configuration": {"stride_fraction": 0.25}}
            )
            bad_timeout = await self.schedule(
                client, {**body, "requested_timeout_seconds": 0}
            )
            provider = await self.schedule(client, {**body, "provider": "standalone"})

        self.assert_error(unknown_builder, 404, "unknown_builder")
        self.assert_error(unknown_model, 404, "unknown_model")
        self.assert_error(incompatible, 422, "incompatible_model")
        for response in (
            empty_key,
            no_key,
            key_in_body,
            settings,
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

    async def test_optional_region_of_interest_may_be_omitted(self) -> None:
        request = {
            **self.schedule_request(),
            "inputs": {"slide": {"kind": "shared_path", "path": "slide.svs"}},
        }
        async with self.client(self.app()) as client:
            response = await self.schedule(client, request)

        self.assertEqual(response.status_code, 202, response.text)

    async def test_scheduled_execution_dispatches_current_model_binding(self) -> None:
        artifact = str(self.root / "heatmap.tiff")

        async def produce(*_args: object) -> BuilderResult:
            return BuilderResult(
                outputs={
                    "heatmap": SharedArtifact(path=artifact, media_type="image/tiff")
                }
            )

        self.dispatch = FakeDispatch(produce)
        self.scheduler = self.make_scheduler(self.repository, self.dispatch)
        async with self.client(self.app()) as client:
            admitted = await self.schedule(client)
            execution_id = admitted.json()["execution"]["id"]
            async with self.scheduler.running(), asyncio.timeout(2):
                while True:
                    state = await client.get(f"/v1/executions/{execution_id}")
                    if state.json()["status"] == "succeeded":
                        break
                    await asyncio.sleep(0.01)

        self.assertEqual(state.json()["result"]["outputs"]["heatmap"]["path"], artifact)
        [(builder_id, invocation, model, context)] = self.dispatch.calls
        self.assertEqual(builder_id, "heatmap")
        self.assertEqual(model, PROSTATE)
        self.assertIsNone(invocation.resolved_configuration)
        self.assertIsNotNone(context.processing_attempt_id)


class RunRouteTests(ApiTestCase):
    async def test_run_returns_result_and_dispatches_without_settings(self) -> None:
        async with self.client(self.app()) as client:
            response = await client.post(RUN, json=self.run_request())

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(), {"outputs": {"score": {"kind": "inline", "value": 0.9}}}
        )
        [(builder_id, invocation, model, context)] = self.dispatch.calls
        self.assertEqual(builder_id, "heatmap")
        self.assertEqual(model, PROSTATE)
        self.assertIsNone(invocation.resolved_configuration)
        self.assertEqual(invocation.output_path, "outputs/heatmap.tiff")
        self.assertIsNone(context.processing_attempt_id)

    async def test_lookup_and_validation_errors_happen_before_dispatch(self) -> None:
        request = self.run_request()
        async with self.client(self.app()) as client:
            responses = {
                "unknown builder": await client.post(
                    "/v1/builders/missing/models/prostate-classifier-1/run",
                    json=request,
                ),
                "unknown model": await client.post(
                    "/v1/builders/heatmap/models/missing/run", json=request
                ),
                "incompatible": await client.post(
                    "/v1/builders/tile-embeddings/models/prostate-classifier-1/run",
                    json=request,
                ),
                "missing input": await client.post(RUN, json={**request, "inputs": {}}),
                "untagged input": await client.post(
                    RUN,
                    json={**request, "inputs": {"slide": {"path": "slide.svs"}}},
                ),
                "past deadline": await client.post(
                    RUN,
                    json={**request, "deadline_at": "2000-01-01T00:00:00+00:00"},
                ),
                "settings": await client.post(
                    RUN,
                    json={**request, "stride_fraction": 0.25},
                ),
            }

        self.assert_error(responses.pop("unknown builder"), 404, "unknown_builder")
        self.assert_error(responses.pop("unknown model"), 404, "unknown_model")
        self.assert_error(responses.pop("incompatible"), 422, "incompatible_model")
        for name, response in responses.items():
            with self.subTest(name):
                self.assert_error(response, 422, "invalid_request")
        self.assertEqual(self.dispatch.calls, [])

    async def test_dispatch_outcomes_map_to_safe_errors(self) -> None:
        def failing(error: Exception) -> Handler:
            async def handler(*_args: object) -> BuilderResult:
                raise error

            return handler

        cases = [
            (
                ConfirmedRejectionError(timedelta(seconds=7)),
                429,
                "builder_backpressure",
                True,
            ),
            (
                RemoteFailureError(
                    code="model_contract_mismatch",
                    safe_message="model changed",
                    retryable=False,
                ),
                422,
                "model_contract_mismatch",
                False,
            ),
            (
                RemoteFailureError(
                    code="model_unavailable", safe_message="down", retryable=True
                ),
                503,
                "model_unavailable",
                True,
            ),
            (UnknownOutcomeError("lost"), 503, "builder_unavailable", True),
        ]
        for error, status_code, code, retryable in cases:
            with self.subTest(code=code):
                self.dispatch = FakeDispatch(failing(error))
                async with self.client(self.app()) as client:
                    response = await client.post(RUN, json=self.run_request())
                self.assert_error(response, status_code, code, retryable=retryable)
                if status_code == 429:
                    self.assertEqual(response.headers["retry-after"], "7")

    async def test_run_is_unavailable_without_a_runner(self) -> None:
        async with self.client(self.app(builder_runner=None)) as client:
            response = await client.post(RUN, json=self.run_request())

        self.assert_error(response, 503, "builder_unavailable", retryable=True)

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
            routes=["/builders/heatmap", "/prostate-classifier-1"],
        )

    async def test_missing_routes_are_separate_from_readiness(self) -> None:
        routes = {"/builders/heatmap/": "heatmap"}
        async with (
            AsyncClient(
                transport=MockTransport(lambda _request: Response(200, json=routes))
            ) as ray,
            self.client(self.app(deployment_health=self.health(ray))) as client,
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
        routes = {"/builders/heatmap": "heatmap", "/prostate-classifier-1": "p"}
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
        async with self.client(self.app(), raise_app_exceptions=False) as client:
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
            async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
                raise SchedulingUnavailableError("temporarily unavailable")

        app = create_app(UnavailableScheduler(), catalog=CATALOG)
        async with self.client(app) as client:
            response = await self.schedule(client)

        self.assert_error(response, 503, "scheduling_unavailable", retryable=True)


class RawInferenceTests(ApiTestCase):
    RUN = "/v1/models/prostate-classifier-1/run"

    def tile(self, size: int = PROSTATE.tile_size) -> bytes:
        return lz4.frame.compress(bytes(size * size * 3))

    def inference_app(self, handler: Callable[[Request], Response]) -> FastAPI:
        ray = AsyncClient(transport=MockTransport(handler))
        self.addAsyncCleanup(ray.aclose)
        proxy = HttpInferenceProxy(ray, origin="http://ray")
        target = ModelInference(CATALOG, proxy, max_result_bytes=1024)
        return self.app(inference_target=target)

    async def test_raw_model_calls_are_forwarded_with_allowed_headers(self) -> None:
        forwarded: list[Request] = []

        def ray(request: Request) -> Response:
            forwarded.append(request)
            return Response(
                200,
                content=b"compressed-result",
                headers={
                    "content-type": "application/octet-stream",
                    "x-output-shape": "(1, 2, 2)",
                    "set-cookie": "session=1",
                },
            )

        tile = self.tile()
        async with self.client(self.inference_app(ray)) as client:
            response = await client.post(
                self.RUN,
                content=tile,
                headers={
                    "content-type": "application/octet-stream",
                    "x-output-dtype": "float32",
                    "authorization": "Bearer facade-only",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, b"compressed-result")
        self.assertEqual(response.headers["x-output-shape"], "(1, 2, 2)")
        self.assertEqual(response.headers["content-type"], "application/octet-stream")
        self.assertNotIn("set-cookie", response.headers)
        [request] = forwarded
        self.assertEqual(request.url, "http://ray/prostate-classifier-1/")
        self.assertEqual(request.content, tile)
        self.assertEqual(request.headers["x-output-dtype"], "float32")
        self.assertNotIn("authorization", request.headers)

    async def test_upstream_status_is_preserved(self) -> None:
        def ray(_request: Request) -> Response:
            return Response(500, content=b"model failed")

        async with self.client(self.inference_app(ray)) as client:
            response = await client.post(self.RUN, content=self.tile())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.content, b"model failed")

    async def test_transport_failure_is_retryable(self) -> None:
        def ray(request: Request) -> Response:
            raise ConnectError("Ray is unavailable", request=request)

        async with self.client(self.inference_app(ray)) as client:
            response = await client.post(self.RUN, content=self.tile())

        self.assert_error(response, 503, "model_inference_unavailable", retryable=True)

    async def test_unavailable_without_an_inference_target(self) -> None:
        async with self.client(self.app()) as client:
            response = await client.post(self.RUN, content=self.tile())

        self.assert_error(response, 503, "model_inference_unavailable", retryable=True)

    async def test_requests_are_checked_before_forwarding(self) -> None:
        forwarded: list[Request] = []

        def ray(request: Request) -> Response:
            forwarded.append(request)
            return Response(200)

        async with self.client(self.inference_app(ray)) as client:
            unknown = await client.post("/v1/models/missing/run", content=self.tile())
            wrong_size = await client.post(self.RUN, content=self.tile(size=224))
            not_lz4 = await client.post(self.RUN, content=b"raw")

        self.assert_error(unknown, 404, "unknown_model")
        self.assert_error(wrong_size, 422, "invalid_request")
        self.assert_error(not_lz4, 422, "invalid_request")
        self.assertEqual(forwarded, [])

    async def test_the_root_proxy_route_is_gone(self) -> None:
        def ray(_request: Request) -> Response:
            return Response(200)

        async with self.client(self.inference_app(ray)) as client:
            response = await client.post("/prostate-classifier-1", content=b"x")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
