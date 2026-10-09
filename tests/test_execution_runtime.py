"""Check production dependency wiring, startup/shutdown, and per-model transport origins."""

from __future__ import annotations

import asyncio
import base64
import unittest
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response

from execution_scheduler.persistence.memory import InMemoryExecutionRepository
from model_service.runtime import create_app
from model_service.settings import SchedulerSettings


class ExecutionRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Exercise production dependency wiring, startup/shutdown, and per-model transport origins."""

    @staticmethod
    def settings() -> SchedulerSettings:
        """Construct production settings with an unused database origin and a synthetic encryption key for wiring tests.

        Returns:
            SchedulerSettings: Parsed and validated scheduler process configuration.
        """
        return SchedulerSettings.from_environment(
            {
                "DATABASE_URL": "postgresql://unused",
                "PROVIDER_CONTEXT_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
                "RAY_SERVE_ORIGIN": "http://workflow-serve",
                "WORKFLOW_IDS": '["prostate-heatmap"]',
            }
        )

    async def test_runtime_only_admits_workflows_enabled_in_this_deployment(self):
        """Reject an undeployed built-in workflow at admission rather than dispatch."""
        environment = {
            "DATABASE_URL": "postgresql://unused",
            "PROVIDER_CONTEXT_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
            "RAY_SERVE_ORIGIN": "http://workflow-serve",
            "WORKFLOW_IDS": '["prostate-heatmap"]',
        }
        app = create_app(SchedulerSettings.from_environment(environment))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://scheduler"
        ) as client:
            response = await client.post(
                "/v1/workflows/gigapath-embeddings/run",
                json={
                    "inputs": {},
                    "resolved_configuration": {},
                    "input_access": {
                        "kind": "shared_storage",
                        "state": {"root": "/tmp"},
                    },
                    "deadline_at": "2099-01-01T00:00:00Z",
                },
            )
        self.assertEqual(response.status_code, 404, response.text)

    async def test_runtime_initializes_repository_and_closes_client_after_scheduler(
        self,
    ):
        """Verify runtime initializes repository and closes client after scheduler.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        repository = InMemoryExecutionRepository()
        initialized = AsyncMock()
        maintained = asyncio.Event()
        tasks_before = asyncio.all_tasks()
        client = AsyncClient(transport=MockTransport(lambda _request: Response(503)))
        with (
            patch.object(repository, "initialize", initialized, create=True),
            patch.object(repository, "maintain", side_effect=maintained.set),
            patch(
                "model_service.runtime.PostgresExecutionRepository",
                return_value=repository,
            ),
            patch("model_service.runtime.httpx.AsyncClient", return_value=client),
        ):
            app = create_app(self.settings())
            async with app.router.lifespan_context(app):
                initialized.assert_awaited_once()
                await asyncio.wait_for(maintained.wait(), timeout=1)
                self.assertFalse(client.is_closed)
            self.assertTrue(client.is_closed)
            self.assertEqual(asyncio.all_tasks(), tasks_before)

    async def test_runtime_closes_client_when_repository_initialization_fails(self):
        """Verify runtime closes client when repository initialization fails.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        repository = InMemoryExecutionRepository()
        initialized = AsyncMock(side_effect=RuntimeError("database unavailable"))
        client = AsyncClient(transport=MockTransport(lambda _request: Response(503)))
        with (
            patch.object(repository, "initialize", initialized, create=True),
            patch(
                "model_service.runtime.PostgresExecutionRepository",
                return_value=repository,
            ),
            patch("model_service.runtime.httpx.AsyncClient", return_value=client),
        ):
            app = create_app(self.settings())
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                async with app.router.lifespan_context(app):
                    self.fail("startup unexpectedly succeeded")
            self.assertTrue(client.is_closed)

    async def test_public_inference_uses_separate_origin_without_changing_workflow_health(
        self,
    ) -> None:
        """Verify public inference uses separate origin without changing workflow health.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        forwarded: list[Request] = []

        def serve_handler(request: Request) -> Response:
            """Return configured route/model responses while recording which upstream origin receives each request.

            Args:
                request: Request to validate, admit, dispatch, forward, or process, as
                    specified by its declared type.

            Returns:
                Response: HTTP response returned by the transport or controlled scenario
                    handler.
            """
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
                "WORKFLOW_IDS": '["prostate-heatmap"]',
            }
        )
        async with AsyncClient(transport=MockTransport(serve_handler)) as serve_client:
            with patch(
                "model_service.runtime.httpx.AsyncClient",
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

    async def test_named_model_shares_workflow_cluster_without_redirecting_other_models(
        self,
    ) -> None:
        """Verify named model shares workflow cluster without redirecting other models.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        forwarded: list[Request] = []

        def serve_handler(request: Request) -> Response:
            """Return configured route/model responses while recording which upstream origin receives each request.

            Args:
                request: Request to validate, admit, dispatch, forward, or process, as
                    specified by its declared type.

            Returns:
                Response: HTTP response returned by the transport or controlled scenario
                    handler.
            """
            forwarded.append(request)
            if request.url.host == "workflow-serve":
                return Response(200, json=0.25)
            if request.url.path == "/prostate-classifier-10":
                return Response(200, json=0.5)
            return Response(404)

        settings = SchedulerSettings.from_environment(
            {
                "DATABASE_URL": "postgresql://unused",
                "PROVIDER_CONTEXT_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
                "RAY_SERVE_ORIGIN": "http://workflow-serve",
                "INFERENCE_SERVE_ORIGIN": "http://model-serve",
                "INFERENCE_MODEL_ORIGINS": '{"prostate-classifier-1":"http://workflow-serve"}',
            }
        )
        async with AsyncClient(transport=MockTransport(serve_handler)) as serve_client:
            with patch(
                "model_service.runtime.httpx.AsyncClient",
                return_value=serve_client,
            ):
                app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://model-service"
            ) as client:
                prediction = await client.post(
                    "/prostate-classifier-1/predict?variant=probe",
                    content=b"compressed-image",
                    headers={"x-output-dtype": "float32"},
                )
                fallback = await client.post(
                    "/prostate-classifier-10", content=b"other"
                )

        self.assertEqual(prediction.status_code, 200)
        self.assertEqual(prediction.json(), 0.25)
        self.assertEqual(fallback.status_code, 200)
        self.assertEqual(fallback.json(), 0.5)
        self.assertEqual(
            forwarded[0].url,
            "http://workflow-serve/prostate-classifier-1/predict?variant=probe",
        )
        self.assertEqual(forwarded[0].content, b"compressed-image")
        self.assertEqual(forwarded[0].headers["x-output-dtype"], "float32")
        self.assertEqual(forwarded[1].url, "http://model-serve/prostate-classifier-10")

    def test_model_origin_configuration_rejects_invalid_prefixes_and_urls(self) -> None:
        """Verify model origin configuration rejects invalid prefixes and urls.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        for origins in (
            '{"prostate-classifier-1":"file:///tmp/model"}',
            '{"prostate-classifier-1":12}',
            '{"prostate-classifier-1/nested":"http://workflow-serve"}',
            "[]",
        ):
            with self.subTest(origins=origins), self.assertRaises(ValueError):
                SchedulerSettings.from_environment(
                    {
                        "DATABASE_URL": "postgresql://unused",
                        "PROVIDER_CONTEXT_KEY": "unused",
                        "RAY_SERVE_ORIGIN": "http://workflow-serve",
                        "INFERENCE_MODEL_ORIGINS": origins,
                    }
                )


if __name__ == "__main__":
    unittest.main()
