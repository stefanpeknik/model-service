from __future__ import annotations

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from httpx import (
    ASGITransport,
    AsyncClient,
    ConnectError,
    MockTransport,
    Request,
    Response,
)
from starlette.types import Message, Scope

from execution_scheduler.contracts import (
    ExecutionStatus,
    ExecutionView,
    ScheduleRequest,
    ScheduleResult,
    SchedulingUnavailableError,
)
from execution_scheduler.persistence.memory import InMemoryExecutionRepository
from execution_scheduler.repository import (
    ExecutionRepositoryUnavailableError,
)
from execution_scheduler.scheduler import ExecutionScheduler
from model_service.api import create_execution_app
from model_service.health import WorkflowDeploymentHealth
from model_service.inference import HttpInferenceProxy
from model_service.synchronous import SynchronousWorkflowService
from tests.test_execution_scheduler import StandaloneProvider
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.contracts import (
    InlineOutput,
    SharedArtifact,
    WorkflowResult,
)
from workflow_execution.dispatch import (
    ConfirmedDispatchRejectionError,
    InMemoryWorkflowDispatch,
)


if TYPE_CHECKING:
    from execution_scheduler.domain import ExecutionRecord


class RecordingScheduler:
    def __init__(self) -> None:
        self.requests: list[ScheduleRequest] = []

    async def check_ready(self) -> None:
        return None

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        self.requests.append(request)
        return ScheduleResult(
            execution=ExecutionView(
                id="execution-1",
                workflow_id=request.workflow_id,
                status=ExecutionStatus.QUEUED,
            ),
            created=True,
        )

    async def get(self, execution_id: str) -> ExecutionView:
        raise AssertionError(execution_id)

    async def cancel(self, execution_id: str) -> ExecutionView:
        raise AssertionError(execution_id)


class ExecutionApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        now = datetime(2026, 9, 17, 8, tzinfo=UTC)
        provider = StandaloneProvider()
        self.scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(clock=lambda: now),
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )
        self.client = AsyncClient(
            transport=ASGITransport(app=create_execution_app(self.scheduler)),
            base_url="http://model-service",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_schedule_replay_poll_and_cancel_use_public_execution_shape(
        self,
    ) -> None:
        request = {
            "provider": "standalone",
            "body": {},
            "idempotency_key": "request-1",
        }

        created = await self.client.post(
            "/v1/workflows/prostate-heatmap/schedule",
            json=request,
        )
        replay = await self.client.post(
            "/v1/workflows/prostate-heatmap/schedule",
            json=request,
        )

        self.assertEqual(created.status_code, 202)
        self.assertEqual(replay.status_code, 200)
        created_body = created.json()
        replay_body = replay.json()
        self.assertTrue(created_body["created"])
        self.assertFalse(replay_body["created"])
        self.assertEqual(
            created_body["execution"]["id"], replay_body["execution"]["id"]
        )
        self.assertEqual(created_body["execution"]["status"], "queued")
        self.assertEqual(
            set(created_body["execution"]),
            {"id", "workflow_id", "status", "result", "error"},
        )

        execution_id = created_body["execution"]["id"]
        polled = await self.client.get(f"/v1/executions/{execution_id}")
        cancelled = await self.client.post(f"/v1/executions/{execution_id}/cancel")
        repeated = await self.client.post(f"/v1/executions/{execution_id}/cancel")

        self.assertEqual(polled.status_code, 200)
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assertEqual(repeated.json(), cancelled.json())

    async def test_empaia_ingress_owns_a_typed_request_shape(self) -> None:
        scheduler = RecordingScheduler()
        async with AsyncClient(
            transport=ASGITransport(app=create_execution_app(scheduler)),
            base_url="http://model-service",
        ) as client:
            response = await client.post(
                "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
                json={
                    "installation_id": "installation-1",
                    "api_url": "https://empaia.example",
                    "job_id": "job-1",
                    "credential": "secret",
                    "requested_timeout_seconds": 600,
                },
            )
            malformed = await client.post(
                "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
                json={
                    "installation_id": "installation-1",
                    "api_url": "https://empaia.example",
                    "job_id": "job-1",
                },
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(malformed.status_code, 422)
        self.assertEqual(len(scheduler.requests), 1)
        request = scheduler.requests[0]
        self.assertEqual(request.provider, "empaia")
        self.assertEqual(request.workflow_id, "prostate-heatmap")
        self.assertEqual(request.requested_timeout_seconds, 600)
        self.assertIsNone(request.idempotency_key)
        self.assertEqual(
            request.body,
            {
                "installation_id": "installation-1",
                "api_url": "https://empaia.example",
                "job_id": "job-1",
                "credential": "secret",
            },
        )

    async def test_scheduled_workflow_completes_through_public_api(self) -> None:
        now = datetime(2026, 9, 17, 8, tzinfo=UTC)
        provider = StandaloneProvider()
        repository = InMemoryExecutionRepository(clock=lambda: now)
        started = asyncio.Event()
        release = asyncio.Event()
        result = WorkflowResult(
            outputs={
                "heatmap": SharedArtifact(
                    path="/shared/heatmap.tiff",
                    media_type="image/tiff",
                )
            }
        )

        async def dispatch(_invocation, _execution_context):
            started.set()
            await release.wait()
            return result

        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
            poll_interval=0.01,
        )
        async with (
            scheduler.running(),
            AsyncClient(
                transport=ASGITransport(app=create_execution_app(scheduler)),
                base_url="http://model-service",
            ) as client,
        ):
            response = await client.post(
                "/v1/workflows/prostate-heatmap/schedule",
                json={
                    "provider": "standalone",
                    "body": {},
                    "idempotency_key": "public-e2e",
                },
            )
            execution_id = response.json()["execution"]["id"]
            await asyncio.wait_for(started.wait(), timeout=1)
            running = await client.get(f"/v1/executions/{execution_id}")
            release.set()
            async with asyncio.timeout(1):
                while True:
                    completed = await client.get(f"/v1/executions/{execution_id}")
                    if completed.json()["status"] == "succeeded":
                        break
                    await asyncio.sleep(0.001)

        self.assertEqual(response.status_code, 202)
        self.assertEqual(running.json()["status"], "running")
        self.assertIsNone(running.json()["result"])
        self.assertEqual(completed.json()["status"], "succeeded")
        self.assertEqual(
            completed.json()["result"]["outputs"]["heatmap"],
            {
                "kind": "shared_artifact",
                "path": "/shared/heatmap.tiff",
                "media_type": "image/tiff",
            },
        )

    async def test_synchronous_workflow_runs_through_the_same_public_origin(
        self,
    ) -> None:
        result = WorkflowResult(outputs={"score": InlineOutput(value=0.9)})
        runner = SynchronousWorkflowService(
            WorkflowCatalog.default(),
            InMemoryWorkflowDispatch(
                {"prostate-heatmap": lambda _invocation, _execution_context: result}
            ),
        )
        async with AsyncClient(
            transport=ASGITransport(
                app=create_execution_app(
                    self.scheduler,
                    workflow_runner=runner,
                )
            ),
            base_url="http://model-service",
        ) as client:
            response = await client.post(
                "/v1/workflows/prostate-heatmap/run",
                json=self._workflow_invocation(),
            )
            unknown = await client.post(
                "/v1/workflows/missing/run",
                json=self._workflow_invocation(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"outputs": {"score": {"kind": "inline", "value": 0.9}}},
        )
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["detail"]["code"], "unknown_workflow")

    async def test_legacy_model_and_slide_calls_use_public_origin(
        self,
    ) -> None:
        forwarded: list[Request] = []

        def ray_handler(request: Request) -> Response:
            forwarded.append(request)
            if request.url.path == "/heatmap-builder":
                return Response(200, text="/mnt/output/heatmap.tiff")
            return Response(
                200,
                content=b"compressed-result",
                headers={
                    "content-type": "application/octet-stream",
                    "x-output-shape": "(1, 2, 2)",
                },
            )

        async with (
            AsyncClient(transport=MockTransport(ray_handler)) as ray_client,
            AsyncClient(
                transport=ASGITransport(
                    app=create_execution_app(
                        self.scheduler,
                        inference_proxy=HttpInferenceProxy(
                            ray_client, origin="http://ray-serve"
                        ),
                    )
                ),
                base_url="http://model-service",
            ) as client,
        ):
            response = await client.post(
                "/prostate-classifier-1?variant=probe",
                content=b"compressed-image",
                headers={
                    "content-type": "application/octet-stream",
                    "x-output-dtype": "float32",
                    "x-pool-tokens": "false",
                    "authorization": "Bearer scheduler-only",
                },
            )
            heatmap = await client.post(
                "/heatmap-builder",
                params={
                    "model_id": "prostate-classifier-1",
                    "slide_path": "/mnt/input/slide.svs",
                    "output_path": "/mnt/output/heatmap.tiff",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"compressed-result")
        self.assertEqual(response.headers["x-output-shape"], "(1, 2, 2)")
        self.assertEqual(len(forwarded), 2)
        self.assertEqual(
            forwarded[0].url, "http://ray-serve/prostate-classifier-1?variant=probe"
        )
        self.assertEqual(forwarded[0].content, b"compressed-image")
        self.assertEqual(
            forwarded[0].headers["content-type"],
            "application/octet-stream",
        )
        self.assertEqual(forwarded[0].headers["x-output-dtype"], "float32")
        self.assertEqual(forwarded[0].headers["x-pool-tokens"], "false")
        self.assertNotIn("authorization", forwarded[0].headers)
        self.assertEqual(heatmap.status_code, 200)
        self.assertEqual(heatmap.text, "/mnt/output/heatmap.tiff")
        self.assertEqual(forwarded[1].url.path, "/heatmap-builder")
        self.assertEqual(forwarded[1].url.params["model_id"], "prostate-classifier-1")
        self.assertEqual(forwarded[1].url.params["slide_path"], "/mnt/input/slide.svs")
        self.assertEqual(
            forwarded[1].url.params["output_path"], "/mnt/output/heatmap.tiff"
        )

        unavailable = await self.client.post(
            "/prostate-classifier-1",
            content=b"compressed-image",
        )
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(
            unavailable.json()["detail"]["code"],
            "model_inference_unavailable",
        )

    async def test_model_inference_resolves_upstream_route_redirect(self) -> None:
        forwarded: list[Request] = []

        def ray_handler(request: Request) -> Response:
            forwarded.append(request)
            if request.url.path == "/prostate-classifier-1":
                return Response(
                    307,
                    headers={"location": "http://ray-serve/prostate-classifier-1/"},
                )
            return Response(200, json=0.25)

        async with (
            AsyncClient(transport=MockTransport(ray_handler)) as ray_client,
            AsyncClient(
                transport=ASGITransport(
                    app=create_execution_app(
                        self.scheduler,
                        inference_proxy=HttpInferenceProxy(
                            ray_client, origin="http://ray-serve"
                        ),
                    )
                ),
                base_url="http://model-service",
            ) as client,
        ):
            response = await client.post(
                "/prostate-classifier-1", content=b"compressed-image"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), 0.25)
        self.assertEqual(len(forwarded), 2)
        self.assertEqual(forwarded[1].url, "http://ray-serve/prostate-classifier-1/")
        self.assertEqual(forwarded[1].content, b"compressed-image")

    async def test_synchronous_workflow_validates_before_dispatch(self) -> None:
        async def reject_dispatch(
            _invocation: object, _execution_context: object
        ) -> WorkflowResult:
            raise AssertionError("invalid invocation was dispatched")

        runner = SynchronousWorkflowService(
            WorkflowCatalog.default(),
            InMemoryWorkflowDispatch({"prostate-heatmap": reject_dispatch}),
        )
        body = self._workflow_invocation()
        body["resolved_configuration"] = {}
        async with AsyncClient(
            transport=ASGITransport(
                app=create_execution_app(
                    self.scheduler,
                    workflow_runner=runner,
                )
            ),
            base_url="http://model-service",
        ) as client:
            response = await client.post(
                "/v1/workflows/prostate-heatmap/run",
                json=body,
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"]["code"],
            "invalid_workflow_invocation",
        )

    async def test_synchronous_workflow_requires_explicit_input_kind(self) -> None:
        runner = SynchronousWorkflowService(
            WorkflowCatalog.default(),
            InMemoryWorkflowDispatch(
                {
                    "prostate-heatmap": lambda _invocation, _execution_context: (
                        WorkflowResult(outputs={})
                    )
                }
            ),
        )
        body = self._workflow_invocation()
        body["inputs"] = {
            "slide": {"path": "slide.svs"},
            "region_of_interest": {"kind": "shared_path", "path": "roi.tiff"},
        }
        async with AsyncClient(
            transport=ASGITransport(
                app=create_execution_app(self.scheduler, workflow_runner=runner)
            ),
            base_url="http://model-service",
        ) as client:
            response = await client.post(
                "/v1/workflows/prostate-heatmap/run", json=body
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"]["code"], "invalid_workflow_invocation"
        )

    async def test_synchronous_workflow_rejects_nonfinite_context_values(self) -> None:
        async def reject_dispatch(
            _invocation: object, _execution_context: object
        ) -> WorkflowResult:
            raise AssertionError("invalid invocation was dispatched")

        runner = SynchronousWorkflowService(
            WorkflowCatalog.default(),
            InMemoryWorkflowDispatch({"prostate-heatmap": reject_dispatch}),
        )
        async with AsyncClient(
            transport=ASGITransport(
                app=create_execution_app(self.scheduler, workflow_runner=runner),
                raise_app_exceptions=False,
            ),
            base_url="http://model-service",
        ) as client:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(value=value):
                    body = self._workflow_invocation()
                    body["input_access"]["state"]["nested"] = [value]
                    response = await client.post(
                        "/v1/workflows/prostate-heatmap/run",
                        content=json.dumps(body),
                        headers={"content-type": "application/json"},
                    )
                    self.assertEqual(response.status_code, 422)
                    self.assertFalse(response.json()["detail"]["retryable"])

    async def test_synchronous_backpressure_preserves_retry_after(self) -> None:
        async def reject_for_capacity(
            _invocation: object, _execution_context: object
        ) -> WorkflowResult:
            raise ConfirmedDispatchRejectionError(timedelta(seconds=7))

        runner = SynchronousWorkflowService(
            WorkflowCatalog.default(),
            InMemoryWorkflowDispatch({"prostate-heatmap": reject_for_capacity}),
        )
        async with AsyncClient(
            transport=ASGITransport(
                app=create_execution_app(
                    self.scheduler,
                    workflow_runner=runner,
                )
            ),
            base_url="http://model-service",
        ) as client:
            response = await client.post(
                "/v1/workflows/prostate-heatmap/run",
                json=self._workflow_invocation(),
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "7")
        self.assertEqual(
            response.json()["detail"]["code"],
            "workflow_backpressure",
        )
        self.assertTrue(response.json()["detail"]["retryable"])

    async def test_synchronous_caller_cancellation_cancels_dispatch(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def wait_for_cancellation(
            _invocation: object, _execution_context: object
        ) -> WorkflowResult:
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("dispatch unexpectedly completed")
            finally:
                cancelled.set()

        runner = SynchronousWorkflowService(
            WorkflowCatalog.default(),
            InMemoryWorkflowDispatch({"prostate-heatmap": wait_for_cancellation}),
        )
        async with AsyncClient(
            transport=ASGITransport(
                app=create_execution_app(
                    self.scheduler,
                    workflow_runner=runner,
                )
            ),
            base_url="http://model-service",
        ) as client:
            request = asyncio.create_task(
                client.post(
                    "/v1/workflows/prostate-heatmap/run",
                    json=self._workflow_invocation(),
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            await asyncio.wait_for(cancelled.wait(), timeout=1)

    async def test_synchronous_asgi_disconnect_cancels_dispatch(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def wait_for_disconnect(
            _invocation: object, _execution_context: object
        ) -> WorkflowResult:
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("dispatch unexpectedly completed")
            finally:
                cancelled.set()

        runner = SynchronousWorkflowService(
            WorkflowCatalog.default(),
            InMemoryWorkflowDispatch({"prostate-heatmap": wait_for_disconnect}),
        )
        app = create_execution_app(self.scheduler, workflow_runner=runner)
        body = json.dumps(self._workflow_invocation()).encode()
        request_sent = False

        async def receive() -> Message:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await started.wait()
            return {"type": "http.disconnect"}

        async def send(_message: Message) -> None:
            pass

        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/workflows/prostate-heatmap/run",
            "raw_path": b"/v1/workflows/prostate-heatmap/run",
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

    async def test_unknown_workflow_and_execution_have_distinct_errors(self) -> None:
        unknown_workflow = await self.client.post(
            "/v1/workflows/missing/schedule",
            json={
                "provider": "standalone",
                "body": {},
                "idempotency_key": "request-1",
            },
        )
        unknown_execution = await self.client.get("/v1/executions/missing")

        self.assertEqual(unknown_workflow.status_code, 404)
        self.assertEqual(unknown_workflow.json()["detail"]["code"], "unknown_workflow")
        self.assertEqual(unknown_execution.status_code, 404)
        self.assertEqual(
            unknown_execution.json()["detail"]["code"], "unknown_execution"
        )

    async def test_missing_application_is_reported_separately_from_readiness(
        self,
    ) -> None:
        async with (
            AsyncClient(
                transport=MockTransport(lambda _request: Response(200, json={}))
            ) as ray_client,
            AsyncClient(
                transport=ASGITransport(
                    app=create_execution_app(
                        self.scheduler,
                        deployment_health=WorkflowDeploymentHealth(
                            ray_client,
                            origin="http://ray",
                            workflow_ids={"prostate-heatmap"},
                        ),
                    )
                ),
                base_url="http://model-service",
            ) as client,
        ):
            live = await client.get("/healthz")
            ready = await client.get("/readyz")
            workflows = await client.get("/healthz/workflows")

        self.assertEqual(live.status_code, 200)
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(workflows.status_code, 503)
        self.assertEqual(workflows.json()["status"], "missing_applications")
        self.assertEqual(
            workflows.json()["missing_workflows"],
            ["prostate-heatmap"],
        )

    async def test_ray_outage_keeps_durable_api_ready_and_usable(self) -> None:
        def unavailable(request: Request) -> Response:
            raise ConnectError("Ray is unavailable", request=request)

        async with AsyncClient(transport=MockTransport(unavailable)) as ray_client:
            app = create_execution_app(
                self.scheduler,
                deployment_health=WorkflowDeploymentHealth(
                    ray_client,
                    origin="http://ray",
                    workflow_ids={"prostate-heatmap"},
                ),
            )
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://model-service"
            ) as client:
                ready = await client.get("/readyz")
                workflows = await client.get("/healthz/workflows")
                admitted = await client.post(
                    "/v1/workflows/prostate-heatmap/schedule",
                    json={
                        "provider": "standalone",
                        "body": {},
                        "idempotency_key": "ray-outage",
                    },
                )
                execution_id = admitted.json()["execution"]["id"]
                polled = await client.get(f"/v1/executions/{execution_id}")
                cancelled = await client.post(f"/v1/executions/{execution_id}/cancel")

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(workflows.status_code, 503)
        self.assertEqual(workflows.json()["status"], "unreachable")
        self.assertEqual(workflows.json()["missing_workflows"], [])
        self.assertEqual(admitted.status_code, 202)
        self.assertEqual(polled.json()["status"], "queued")
        self.assertEqual(cancelled.json()["status"], "cancelled")

    async def test_database_outage_fails_durable_readiness(self) -> None:
        class UnavailableRepository(InMemoryExecutionRepository):
            async def check_ready(self) -> None:
                raise ExecutionRepositoryUnavailableError("database unavailable")

        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=UnavailableRepository(),
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
        )
        async with AsyncClient(
            transport=ASGITransport(app=create_execution_app(scheduler)),
            base_url="http://model-service",
        ) as client:
            ready = await client.get("/readyz")
            live = await client.get("/healthz")

        self.assertEqual(ready.status_code, 503)
        self.assertEqual(ready.json()["detail"]["code"], "scheduling_unavailable")
        self.assertEqual(live.status_code, 200)

    async def test_database_outage_makes_polling_and_cancellation_retryable(
        self,
    ) -> None:
        class UnavailableRepository(InMemoryExecutionRepository):
            async def get(self, execution_id: str) -> ExecutionRecord:
                raise ExecutionRepositoryUnavailableError(execution_id)

            async def request_cancel(self, execution_id: str) -> ExecutionRecord:
                raise ExecutionRepositoryUnavailableError(execution_id)

        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=UnavailableRepository(),
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
        )
        async with AsyncClient(
            transport=ASGITransport(
                app=create_execution_app(scheduler), raise_app_exceptions=False
            ),
            base_url="http://model-service",
        ) as client:
            responses = [
                await client.get("/v1/executions/execution-1"),
                await client.post("/v1/executions/execution-1/cancel"),
            ]

        for response in responses:
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["detail"]["code"], "scheduling_unavailable"
            )
            self.assertTrue(response.json()["detail"]["retryable"])

    async def test_temporary_scheduling_unavailability_is_retryable(self) -> None:
        class UnavailableScheduler:
            async def check_ready(self) -> None:
                raise SchedulingUnavailableError("repository unavailable")

            async def schedule(self, _request: ScheduleRequest) -> ScheduleResult:
                raise SchedulingUnavailableError(
                    "execution scheduling is temporarily unavailable"
                )

            async def get(self, _execution_id: str) -> ExecutionView:
                raise AssertionError

            async def cancel(self, _execution_id: str) -> ExecutionView:
                raise AssertionError

        async with AsyncClient(
            transport=ASGITransport(
                app=create_execution_app(UnavailableScheduler()),
            ),
            base_url="http://model-service",
        ) as client:
            response = await client.post(
                "/v1/workflows/prostate-heatmap/schedule",
                json={
                    "provider": "standalone",
                    "body": {},
                    "idempotency_key": "request-1",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "scheduling_unavailable")
        self.assertTrue(response.json()["detail"]["retryable"])

    @staticmethod
    def _workflow_invocation() -> dict[str, object]:
        return {
            "inputs": {
                "slide": {"kind": "shared_path", "path": "slide.svs"},
                "region_of_interest": {
                    "kind": "shared_path",
                    "path": "roi.tiff",
                },
            },
            "resolved_configuration": {
                "output_path": "outputs/heatmap.tiff",
                "stride_fraction": 0.5,
                "output_tile_height": 256,
                "output_tile_width": 256,
            },
            "input_access": {
                "kind": "shared_storage",
                "state": {"root": "/shared"},
            },
            "deadline_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        }


if __name__ == "__main__":
    unittest.main()
