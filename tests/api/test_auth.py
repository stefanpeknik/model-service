"""Authentication and authorization plug in per API; by default everyone is let in."""

from __future__ import annotations

import unittest

from fastapi import Request
from httpx import ASGITransport, AsyncClient

from model_service.api.app_factory import create_app
from model_service.contracts.model_declarations import SpatialScoresModel
from model_service.deployed_models.catalog import ModelCatalog
from model_service.http_common.auth import (
    Access,
    Action,
    AuthenticationError,
    AuthorizationError,
    Principal,
)


CATALOG = ModelCatalog(
    [
        SpatialScoresModel(
            id="classifier", tile_size=8, output_tile_size=1, n_channels=1, mpp=0.5
        )
    ]
)
RUN = "/v1/builders/heatmap/models/classifier/run"


class BearerToken:
    """Accepts one token and names its holder."""

    async def authenticate(self, request: Request) -> Principal:
        if request.headers.get("authorization") != "Bearer student-token":
            raise AuthenticationError
        return Principal("student")


class Recorder:
    """Refuses runs and records every decision it was asked for."""

    def __init__(self) -> None:
        self.asked: list[tuple[str | None, Action]] = []

    async def authorize(self, principal: Principal, action: Action) -> None:
        self.asked.append((principal.subject, action))
        if action is Action.RUN:
            raise AuthorizationError


def client(**access: Access) -> AsyncClient:
    app = create_app(catalog=CATALOG, **access)
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

    async def test_the_authorizer_decides_each_action(self) -> None:
        authorizer = Recorder()
        access = Access(authenticator=BearerToken(), authorizer=authorizer)
        headers = {"Authorization": "Bearer student-token"}
        async with client(access=access) as http:
            read = await http.get("/v1/builders", headers=headers)
            run = await http.post(RUN, headers=headers, json={})

        self.assertEqual(read.status_code, 200)
        self.assertEqual(run.status_code, 403)
        self.assertEqual(run.json()["detail"]["code"], "forbidden")
        self.assertEqual(
            authorizer.asked,
            [("student", Action.READ_CATALOG), ("student", Action.RUN)],
        )


if __name__ == "__main__":
    unittest.main()
