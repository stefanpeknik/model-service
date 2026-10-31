"""Authentication and authorization plug in per API; by default everyone is let in."""

from __future__ import annotations

import unittest

from fastapi import Request
from httpx import ASGITransport, AsyncClient

from model_service.builders.contracts import SpatialScoresModel
from model_service.facade.api import create_app
from model_service.facade.auth import (
    Access,
    Action,
    AuthenticationError,
    AuthorizationError,
    Principal,
)
from model_service.facade.catalog import ModelCatalog
from model_service.scheduler.contracts import (
    ExecutionStatus,
    ExecutionView,
    Operation,
)


CATALOG = ModelCatalog(
    [
        SpatialScoresModel(
            id="classifier", tile_size=8, output_tile_size=1, n_channels=1, mpp=0.5
        )
    ]
)
EXECUTION = "00000000-0000-4000-8000-000000000001"


class Scheduler:
    async def check_ready(self) -> None:
        return None

    async def get(self, execution_id: str) -> ExecutionView:
        return ExecutionView(
            id=execution_id,
            operation=Operation.BUILDER,
            builder_id="heatmap",
            model_id="classifier",
            status=ExecutionStatus.QUEUED,
        )


class BearerToken:
    """Accepts one token and names its holder."""

    async def authenticate(self, request: Request) -> Principal:
        if request.headers.get("authorization") != "Bearer student-token":
            raise AuthenticationError
        return Principal("student")


class Recorder:
    """Refuses cancellation and records every decision it was asked for."""

    def __init__(self) -> None:
        self.asked: list[tuple[str | None, Action, str | None]] = []

    async def authorize(
        self, principal: Principal, action: Action, execution_id: str | None
    ) -> None:
        self.asked.append((principal.subject, action, execution_id))
        if action is Action.CANCEL_EXECUTION:
            raise AuthorizationError


def client(**access: Access) -> AsyncClient:
    app = create_app(Scheduler(), catalog=CATALOG, **access)  # type: ignore[arg-type]
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://ms")


class AuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_everyone_is_let_in_by_default(self) -> None:
        async with client() as http:
            response = await http.get("/v1/models")

        self.assertEqual(response.status_code, 200)

    async def test_an_authenticator_rejects_requests_without_credentials(
        self,
    ) -> None:
        async with client(access=Access(authenticator=BearerToken())) as http:
            anonymous = await http.get("/v1/models")
            student = await http.get(
                "/v1/models", headers={"Authorization": "Bearer student-token"}
            )
            health = await http.get("/healthz")

        self.assertEqual(anonymous.status_code, 401)
        self.assertEqual(anonymous.json()["detail"]["code"], "unauthenticated")
        self.assertEqual(anonymous.headers["www-authenticate"], "Bearer")
        self.assertEqual(student.status_code, 200)
        self.assertEqual(health.status_code, 200)

    async def test_the_authorizer_decides_each_action_on_its_execution(self) -> None:
        authorizer = Recorder()
        access = Access(authenticator=BearerToken(), authorizer=authorizer)
        headers = {"Authorization": "Bearer student-token"}
        async with client(access=access) as http:
            read = await http.get(f"/v1/executions/{EXECUTION}", headers=headers)
            cancel = await http.post(
                f"/v1/executions/{EXECUTION}/cancel", headers=headers
            )

        self.assertEqual(read.status_code, 200)
        self.assertEqual(cancel.status_code, 403)
        self.assertEqual(cancel.json()["detail"]["code"], "forbidden")
        self.assertEqual(
            authorizer.asked,
            [
                ("student", Action.READ_EXECUTION, EXECUTION),
                ("student", Action.CANCEL_EXECUTION, EXECUTION),
            ],
        )

    async def test_the_empaia_route_has_its_own_access(self) -> None:
        async with client(access=Access(authenticator=BearerToken())) as http:
            empaia = await http.get(f"/v1/empaia/executions/{EXECUTION}")
            public = await http.get(f"/v1/executions/{EXECUTION}")

        self.assertEqual(empaia.status_code, 200)
        self.assertEqual(public.status_code, 401)


if __name__ == "__main__":
    unittest.main()
