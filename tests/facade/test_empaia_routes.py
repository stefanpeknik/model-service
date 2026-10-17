"""The JES remote execution protocol at `/v1/empaia` and the EMPAIA app catalog."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from httpx import ASGITransport, AsyncClient

from model_service.builders.contracts import SpatialScoresModel
from model_service.facade.api import create_app
from model_service.facade.catalog import ModelCatalog
from model_service.providers.empaia.apps import EmpaiaApp, EmpaiaApps
from model_service.scheduler.contracts import (
    ExecutionError,
    ExecutionStatus,
    ExecutionView,
    Operation,
    ScheduleRequest,
    ScheduleResult,
    UnknownExecutionError,
)


ROOT = Path(__file__).resolve().parents[2]
PROSTATE = SpatialScoresModel(
    id="prostate-classifier-1",
    tile_size=512,
    output_tile_size=1,
    n_channels=1,
    mpp=0.468,
)
APPS = EmpaiaApps(
    {"prostate-heatmap": EmpaiaApp(builder="heatmap", model="prostate-classifier-1")}
)
TOKEN = "secret-job-token"
JOB = {
    "app_api": "https://app-service.example/v3/",
    "job_id": "job-1",
    "access_token": TOKEN,
    "env": {"SERVICE_NAME": "ignored"},
}


def view(status: ExecutionStatus, **fields: object) -> ExecutionView:
    return ExecutionView(
        id="execution-1",
        operation=Operation.BUILDER,
        builder_id="heatmap",
        model_id=PROSTATE.id,
        status=status,
        **fields,  # type: ignore[arg-type]
    )


class ScriptedScheduler:
    """Admits requests and answers status and cancellation from a fixed view."""

    def __init__(self, current: ExecutionView, *, created: bool = True) -> None:
        self.current = current
        self.created = created
        self.requests: list[ScheduleRequest] = []
        self.cancelled: list[str] = []

    async def check_ready(self) -> None:
        return None

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        self.requests.append(request)
        return ScheduleResult(self.current, created=self.created)

    async def get(self, execution_id: str) -> ExecutionView:
        if execution_id != self.current.id:
            raise UnknownExecutionError(execution_id)
        return self.current

    async def cancel(self, execution_id: str) -> ExecutionView:
        self.cancelled.append(execution_id)
        return await self.get(execution_id)


def client(scheduler: ScriptedScheduler, apps: EmpaiaApps | None = APPS) -> AsyncClient:
    app = create_app(
        scheduler,  # type: ignore[arg-type]
        catalog=ModelCatalog([PROSTATE]),
        empaia_apps=apps,
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://ms")


class EmpaiaRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_submission_runs_the_apps_builder_and_model(self) -> None:
        scheduler = ScriptedScheduler(view(ExecutionStatus.QUEUED))
        async with client(scheduler) as http:
            response = await http.post(
                "/v1/empaia/apps/prostate-heatmap",
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
            response = await http.post("/v1/empaia/apps/prostate-heatmap", json=JOB)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["location"], "/v1/empaia/executions/execution-1"
        )

    async def test_unknown_app_and_invalid_body_are_rejected_safely(self) -> None:
        scheduler = ScriptedScheduler(view(ExecutionStatus.QUEUED))
        async with client(scheduler) as http:
            unknown = await http.post("/v1/empaia/apps/other-app", json=JOB)
            invalid = await http.post(
                "/v1/empaia/apps/prostate-heatmap", json={**JOB, "job_id": ""}
            )

        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["detail"]["code"], "unknown_app")
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

    async def test_routes_exist_only_when_empaia_is_configured(self) -> None:
        scheduler = ScriptedScheduler(view(ExecutionStatus.QUEUED))
        async with client(scheduler, apps=None) as http:
            response = await http.post("/v1/empaia/apps/prostate-heatmap", json=JOB)

        self.assertNotEqual(response.status_code, 202)
        self.assertEqual(scheduler.requests, [])


class EmpaiaAppsTests(unittest.TestCase):
    def test_rendered_catalog_is_read_and_unknown_builders_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empaia-apps.json"
            path.write_text(
                json.dumps(
                    {
                        "prostate-heatmap": {
                            "builder": "heatmap",
                            "model": "prostate-classifier-1",
                            "empaia": {"name": "ignored here"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            apps = EmpaiaApps.from_file(path)
            path.write_text(
                json.dumps({"broken": {"builder": "nonexistent", "model": "m"}}),
                encoding="utf-8",
            )
            with self.assertRaises(KeyError):
                EmpaiaApps.from_file(path)
            path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "keyed by app ID"):
                EmpaiaApps.from_file(path)

        self.assertEqual(apps.get("prostate-heatmap").model, "prostate-classifier-1")

    def test_repository_app_files_are_valid_apps(self) -> None:
        directory = ROOT / "deploy" / "helm" / "rayservice" / "empaia-apps"
        apps = EmpaiaApps(
            {
                path.stem: EmpaiaApp.model_validate(yaml.safe_load(path.read_text()))
                for path in directory.glob("*.yaml")
            }
        )
        self.assertEqual(apps.get("prostate-heatmap").builder, "heatmap")


if __name__ == "__main__":
    unittest.main()
