from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from httpx import ASGITransport, AsyncClient

from execution_scheduler.api import create_execution_app
from execution_scheduler.contracts import (
    ExecutionView,
    ScheduleRequest,
    ScheduleResult,
    SchedulingUnavailableError,
)
from execution_scheduler.dispatch import InMemoryWorkflowDispatch
from execution_scheduler.repository import InMemoryExecutionRepository
from execution_scheduler.scheduler import ExecutionScheduler, ExecutionWorker
from tests.test_execution_scheduler import StandaloneProvider
from workflow_execution.contracts import SharedArtifact, WorkflowResult


class ExecutionApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        now = datetime(2026, 9, 17, 8, tzinfo=UTC)
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(clock=lambda: now),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )
        self.client = AsyncClient(
            transport=ASGITransport(app=create_execution_app(scheduler)),
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

    async def test_scheduled_workflow_completes_through_public_api(self) -> None:
        now = datetime(2026, 9, 17, 8, tzinfo=UTC)
        provider = StandaloneProvider()
        repository = InMemoryExecutionRepository(clock=lambda: now)
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )
        result = WorkflowResult(
            outputs={
                "heatmap": SharedArtifact(
                    path="/shared/heatmap.tiff",
                    media_type="image/tiff",
                )
            }
        )
        worker = ExecutionWorker(
            repository=repository,
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch(
                {"prostate-heatmap": lambda _invocation: result}
            ),
            owner="worker-1",
            lease_duration=timedelta(minutes=1),
        )
        async with AsyncClient(
            transport=ASGITransport(app=create_execution_app(scheduler)),
            base_url="http://model-service",
        ) as client:
            response = await client.post(
                "/v1/workflows/prostate-heatmap/schedule",
                json={
                    "provider": "standalone",
                    "body": {},
                    "idempotency_key": "public-e2e",
                },
            )
            execution_id = response.json()["execution"]["id"]
            await worker.run_once()
            running = await client.get(f"/v1/executions/{execution_id}")
            await worker.run_once()
            completed = await client.get(f"/v1/executions/{execution_id}")

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

    async def test_readiness_reports_declared_but_missing_application(self) -> None:
        class MissingDeployment:
            async def missing_workflows(self) -> tuple[str, ...]:
                return ("prostate-heatmap",)

        now = datetime(2026, 9, 17, 8, tzinfo=UTC)
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(clock=lambda: now),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )
        client = AsyncClient(
            transport=ASGITransport(
                app=create_execution_app(
                    scheduler,
                    deployment_health=MissingDeployment(),
                )
            ),
            base_url="http://model-service",
        )
        async with client:
            live = await client.get("/healthz")
            ready = await client.get("/readyz")

        self.assertEqual(live.status_code, 200)
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(
            ready.json()["detail"]["code"],
            "workflow_application_unavailable",
        )
        self.assertEqual(
            ready.json()["detail"]["workflows"],
            ["prostate-heatmap"],
        )

    async def test_temporary_scheduling_unavailability_is_retryable(self) -> None:
        class UnavailableScheduler:
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


if __name__ == "__main__":
    unittest.main()
