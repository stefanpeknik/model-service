"""Running executions explain when Ray Serve cannot place their replicas."""

from __future__ import annotations

import unittest

from httpx import (
    ASGITransport,
    AsyncClient,
    ConnectError,
    MockTransport,
    Request,
    Response,
)

from model_service.facade.api import create_app
from model_service.facade.capacity import ServeCapacity
from model_service.facade.catalog import ModelCatalog
from model_service.scheduler.contracts import ExecutionStatus
from tests.facade.test_empaia_routes import PROSTATE, ScriptedScheduler, view


NOW = 10_000.0
EMPAIA_STATUS = "/v1/empaia/executions/execution-1"


def replica(
    state: str, started: float, node_id: str | None = None
) -> dict[str, object]:
    return {"state": state, "start_time_s": started, "node_id": node_id}


def application(*replicas: dict[str, object]) -> dict[str, object]:
    return {"deployments": {"Model": {"replicas": list(replicas)}}}


class FakeDashboard:
    def __init__(self, applications: dict[str, object]) -> None:
        self.applications = applications
        self.requests = 0

    def __call__(self, request: Request) -> Response:
        self.requests += 1
        assert request.url.path == "/api/serve/applications/"
        return Response(200, json={"applications": self.applications})


class CapacityMessageTests(unittest.IsolatedAsyncioTestCase):
    async def status(
        self,
        dashboard: object,
        status: ExecutionStatus = ExecutionStatus.RUNNING,
        path: str = EMPAIA_STATUS,
    ) -> dict[str, object]:
        async with AsyncClient(transport=MockTransport(dashboard)) as ray:  # type: ignore[arg-type]
            capacity = ServeCapacity(
                ray, origin="http://ray-head:8265", clock=lambda: NOW
            )
            app = create_app(
                ScriptedScheduler(view(status)),  # type: ignore[arg-type]
                catalog=ModelCatalog([PROSTATE]),
                empaia_enabled=True,
                capacity=capacity,
            )
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://ms"
            ) as http:
                response = await http.get(path)
        self.assertEqual(response.status_code, 200)
        return response.json()

    async def test_unplaced_model_replica_is_explained_after_the_grace_period(
        self,
    ) -> None:
        dashboard = FakeDashboard(
            {PROSTATE.id: application(replica("STARTING", NOW - 300))}
        )

        empaia = await self.status(dashboard)
        execution = await self.status(dashboard, path="/v1/executions/execution-1")

        message = (
            f"waiting for capacity: no replica of model {PROSTATE.id} "
            "could be scheduled for 300s"
        )
        self.assertEqual(empaia, {"status": "running", "message": message})
        self.assertEqual(execution["status"], "running")
        self.assertEqual(execution["message"], message)

    async def test_unplaced_builder_replica_names_the_builder(self) -> None:
        dashboard = FakeDashboard(
            {"builder-heatmap": application(replica("STARTING", NOW - 90))}
        )

        self.assertEqual(
            (await self.status(dashboard))["message"],
            "waiting for capacity: no replica of builder heatmap "
            "could be scheduled for 90s",
        )

    async def test_normal_startup_and_service_are_not_explained(self) -> None:
        cases = {
            "within grace": application(replica("STARTING", NOW - 30)),
            "placed and loading": application(
                replica("STARTING", NOW - 600, node_id="node-1")
            ),
            "another replica serves": application(
                replica("RUNNING", NOW - 900, node_id="node-1"),
                replica("STARTING", NOW - 600),
            ),
        }
        for case, deployed in cases.items():
            with self.subTest(case):
                body = await self.status(FakeDashboard({PROSTATE.id: deployed}))
                self.assertEqual(body, {"status": "running", "message": None})

    async def test_only_running_executions_ask_ray(self) -> None:
        dashboard = FakeDashboard(
            {PROSTATE.id: application(replica("STARTING", NOW - 300))}
        )

        body = await self.status(dashboard, status=ExecutionStatus.QUEUED)

        self.assertEqual(body, {"status": "queued", "message": None})
        self.assertEqual(dashboard.requests, 0)

    async def test_unreachable_dashboard_leaves_the_status_alone(self) -> None:
        def unreachable(request: Request) -> Response:
            raise ConnectError("dashboard is down", request=request)

        body = await self.status(unreachable)

        self.assertEqual(body, {"status": "running", "message": None})

    async def test_polls_within_the_cache_window_share_one_dashboard_read(
        self,
    ) -> None:
        dashboard = FakeDashboard({})
        async with AsyncClient(transport=MockTransport(dashboard)) as ray:
            capacity = ServeCapacity(
                ray, origin="http://ray-head:8265", clock=lambda: NOW
            )
            for _ in range(3):
                await capacity.explain(view(ExecutionStatus.RUNNING))

        self.assertEqual(dashboard.requests, 1)


if __name__ == "__main__":
    unittest.main()
