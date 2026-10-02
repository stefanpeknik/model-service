from __future__ import annotations

import asyncio
import base64
import unittest
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response

from execution_scheduler.repository import InMemoryExecutionRepository
from execution_scheduler.runtime import create_app
from execution_scheduler.settings import SchedulerSettings


class ExecutionRuntimeTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def settings() -> SchedulerSettings:
        return SchedulerSettings.from_environment(
            {
                "DATABASE_URL": "postgresql://unused",
                "PROVIDER_CONTEXT_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
                "RAY_SERVE_ORIGIN": "http://workflow-serve",
            }
        )

    async def test_runtime_initializes_repository_and_closes_client_after_scheduler(
        self,
    ):
        repository = InMemoryExecutionRepository()
        initialized = AsyncMock()
        maintained = asyncio.Event()
        tasks_before = asyncio.all_tasks()
        client = AsyncClient(transport=MockTransport(lambda _request: Response(503)))
        with (
            patch.object(repository, "initialize", initialized, create=True),
            patch.object(repository, "maintain", side_effect=maintained.set),
            patch(
                "execution_scheduler.runtime.PostgresExecutionRepository",
                return_value=repository,
            ),
            patch("execution_scheduler.runtime.httpx.AsyncClient", return_value=client),
        ):
            app = create_app(self.settings())
            async with app.router.lifespan_context(app):
                initialized.assert_awaited_once()
                await asyncio.wait_for(maintained.wait(), timeout=1)
                self.assertFalse(client.is_closed)
            self.assertTrue(client.is_closed)
            self.assertEqual(asyncio.all_tasks(), tasks_before)

    async def test_runtime_closes_client_when_repository_initialization_fails(self):
        repository = InMemoryExecutionRepository()
        initialized = AsyncMock(side_effect=RuntimeError("database unavailable"))
        client = AsyncClient(transport=MockTransport(lambda _request: Response(503)))
        with (
            patch.object(repository, "initialize", initialized, create=True),
            patch(
                "execution_scheduler.runtime.PostgresExecutionRepository",
                return_value=repository,
            ),
            patch("execution_scheduler.runtime.httpx.AsyncClient", return_value=client),
        ):
            app = create_app(self.settings())
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                async with app.router.lifespan_context(app):
                    self.fail("startup unexpectedly succeeded")
            self.assertTrue(client.is_closed)

    async def test_public_inference_uses_separate_origin_without_changing_workflow_health(
        self,
    ) -> None:
        forwarded: list[Request] = []

        def serve_handler(request: Request) -> Response:
            forwarded.append(request)
            if request.url == "http://workflow-serve/-/routes":
                return Response(
                    200, json={"/workflows/prostate-heatmap": "prostate-heatmap"}
                )
            if request.url == "http://model-serve/prostate-classifier-1":
                return Response(200, json=0.25)
            return Response(404)

        settings = SchedulerSettings.from_environment(
            {
                "DATABASE_URL": "postgresql://unused",
                "PROVIDER_CONTEXT_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
                "RAY_SERVE_ORIGIN": "http://workflow-serve",
                "INFERENCE_SERVE_ORIGIN": "http://model-serve",
            }
        )
        async with AsyncClient(transport=MockTransport(serve_handler)) as serve_client:
            with patch(
                "execution_scheduler.runtime.httpx.AsyncClient",
                return_value=serve_client,
            ):
                app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://model-service",
            ) as client:
                workflows = await client.get("/healthz/workflows")
                prediction = await client.post(
                    "/prostate-classifier-1", content=b"compressed-image"
                )

        self.assertEqual(workflows.status_code, 200)
        self.assertEqual(prediction.status_code, 200)
        self.assertEqual(prediction.json(), 0.25)
        self.assertEqual(forwarded[0].url, "http://workflow-serve/-/routes")
        self.assertEqual(forwarded[1].url, "http://model-serve/prostate-classifier-1")
        self.assertEqual(forwarded[1].content, b"compressed-image")
        self.assertEqual(forwarded[1].extensions["timeout"]["read"], 900)


if __name__ == "__main__":
    unittest.main()
