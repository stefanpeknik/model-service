"""The JES remote execution protocol at `/v1/empaia`."""

from __future__ import annotations

import unittest

from fastapi import Request
from httpx import ASGITransport, AsyncClient

from model_service.facade.api import create_app
from model_service.facade.auth import Access, AuthenticationError, Principal
from model_service.facade.catalog import ModelCatalog
from model_service.providers.empaia.routes import create_router
from model_service.scheduler.contracts import (
    ExecutionError,
    ExecutionStatus,
    ScheduleRequest,
)
from tests.support.scheduling import PROSTATE, ScriptedScheduler, view


SUBMIT = "/v1/empaia/builders/heatmap/models/prostate-classifier-1"
TOKEN = "secret-job-token"
JOB = {
    "app_api": "https://app-service.example/v3/",
    "job_id": "job-1",
    "access_token": TOKEN,
    "env": {"SERVICE_NAME": "ignored"},
}


def client(scheduler: ScriptedScheduler) -> AsyncClient:
    app = create_app(
        scheduler,  # type: ignore[arg-type]
        catalog=ModelCatalog([PROSTATE]),
        provider_routes=[create_router],
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://ms")


class EmpaiaRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_submission_runs_the_apps_builder_and_model(self) -> None:
        scheduler = ScriptedScheduler(view(ExecutionStatus.QUEUED))
        async with client(scheduler) as http:
            response = await http.post(
                SUBMIT,
                json=JOB,
                headers={"Idempotency-Key": "job-1"},
            )

        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(
            response.headers["location"], "/v1/empaia/executions/execution-1"
        )
        self.assertEqual(response.json(), {"status": "queued", "message": None})
        self.assertEqual(
            scheduler.requests,
            [
                ScheduleRequest(
                    builder_id="heatmap",
                    model_id=PROSTATE.id,
                    provider="empaia",
                    body={
                        "installation_id": "https://app-service.example/v3",
                        "api_url": "https://app-service.example/v3/",
                        "job_id": "job-1",
                        "credential": TOKEN,
                    },
                )
            ],
        )

    async def test_resubmission_returns_the_same_execution(self) -> None:
        scheduler = ScriptedScheduler(view(ExecutionStatus.RUNNING), created=False)
        async with client(scheduler) as http:
            response = await http.post(SUBMIT, json=JOB)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["location"], "/v1/empaia/executions/execution-1"
        )

    async def test_idempotency_key_must_be_the_job_id(self) -> None:
        scheduler = ScriptedScheduler(view(ExecutionStatus.QUEUED))
        async with client(scheduler) as http:
            response = await http.post(
                SUBMIT, json=JOB, headers={"Idempotency-Key": "another-job"}
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "invalid_request")
        self.assertEqual(scheduler.requests, [])

    async def test_unknown_model_and_invalid_body_are_rejected_safely(self) -> None:
        scheduler = ScriptedScheduler(view(ExecutionStatus.QUEUED))
        scheduler.unknown_models.add("other-model")
        async with client(scheduler) as http:
            unknown = await http.post(
                "/v1/empaia/builders/heatmap/models/other-model", json=JOB
            )
            invalid = await http.post(SUBMIT, json={**JOB, "job_id": ""})

        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["detail"]["code"], "unknown_model")
        self.assertEqual(invalid.status_code, 422)
        self.assertNotIn(TOKEN, invalid.text)
        self.assertEqual(scheduler.requests, [])

    async def test_status_reports_a_failure_only_after_empaia_was_told(self) -> None:
        error = ExecutionError("model_failed", "the model failed")
        cases = [
            (view(ExecutionStatus.RUNNING), {"status": "running", "message": None}),
            (
                view(ExecutionStatus.SUCCEEDED),
                {"status": "succeeded", "message": None},
            ),
            (
                view(ExecutionStatus.FAILED, error=error, provider_pending=True),
                {"status": "running", "message": None},
            ),
            (
                view(ExecutionStatus.FAILED, error=error),
                {"status": "failed", "message": "the model failed"},
            ),
            (
                view(ExecutionStatus.CANCELLED, provider_pending=True),
                {"status": "running", "message": None},
            ),
            (
                view(ExecutionStatus.CANCELLED),
                {"status": "cancelled", "message": None},
            ),
        ]
        for current, expected in cases:
            with self.subTest(status=current.status, pending=current.provider_pending):
                async with client(ScriptedScheduler(current)) as http:
                    response = await http.get("/v1/empaia/executions/execution-1")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), expected)

    async def test_delete_cancels_and_unknown_executions_are_404(self) -> None:
        scheduler = ScriptedScheduler(view(ExecutionStatus.CANCELLED))
        async with client(scheduler) as http:
            cancelled = await http.delete("/v1/empaia/executions/execution-1")
            unknown_get = await http.get("/v1/empaia/executions/other")
            unknown_delete = await http.delete("/v1/empaia/executions/other")

        self.assertEqual(cancelled.json(), {"status": "cancelled", "message": None})
        self.assertEqual(scheduler.cancelled, ["execution-1", "other"])
        self.assertEqual(unknown_get.status_code, 404)
        self.assertEqual(unknown_delete.status_code, 404)

    async def test_the_empaia_routes_have_their_own_access(self) -> None:
        class Nobody:
            async def authenticate(self, request: Request) -> Principal:
                raise AuthenticationError

        app = create_app(
            ScriptedScheduler(view(ExecutionStatus.QUEUED)),  # type: ignore[arg-type]
            catalog=ModelCatalog([PROSTATE]),
            access=Access(authenticator=Nobody()),
            provider_routes=[create_router],
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://ms"
        ) as http:
            empaia = await http.get("/v1/empaia/executions/execution-1")
            public = await http.get("/v1/executions/execution-1")

        self.assertEqual(empaia.status_code, 200)
        self.assertEqual(public.status_code, 401)


if __name__ == "__main__":
    unittest.main()
