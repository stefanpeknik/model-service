"""Check production wiring: catalog loading, builder selection and lifecycle."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response

from model_service.api.app_factory import build_api_process
from model_service.deployed_models.catalog import CatalogError
from model_service.scheduler.settings import SchedulerSettings
from tests.support.model_documents import embedding_document, spatial_document


class RuntimeTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.catalog_path = self.directory / "model-catalog.json"
        self.catalog_path.write_text(
            json.dumps([spatial_document(), embedding_document()]), encoding="utf-8"
        )

    def settings(self, **environment: str) -> SchedulerSettings:
        values = {
            "RAY_SERVE_ORIGIN": "http://builder-serve",
            "MODEL_CATALOG_PATH": str(self.catalog_path),
            **environment,
        }
        with patch.dict(os.environ, values, clear=True):
            return SchedulerSettings()


class ExecutionRuntimeTests(RuntimeTestCase):
    async def test_construction_loads_catalog(self) -> None:
        app = build_api_process(self.settings())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://model-service"
        ) as client:
            models = await client.get("/v1/models")

        self.assertEqual(
            [model["id"] for model in models.json()],
            ["prostate-classifier-1", "virchow2"],
        )

    def test_invalid_catalog_or_builder_selection_prevents_startup(self) -> None:
        self.catalog_path.write_text(
            json.dumps([spatial_document(), spatial_document()]), encoding="utf-8"
        )
        with self.assertRaises(CatalogError):
            build_api_process(self.settings())
        self.catalog_path.write_text("[]", encoding="utf-8")
        with self.assertRaises(KeyError):
            build_api_process(self.settings(BUILDER_IDS='["missing"]'))

    async def test_lifespan_closes_client(self) -> None:
        client = AsyncClient(transport=MockTransport(lambda _request: Response(503)))
        with patch(
            "model_service.api.app_factory.httpx.AsyncClient", return_value=client
        ):
            app = build_api_process(self.settings())
            async with app.router.lifespan_context(app):
                self.assertFalse(client.is_closed)
            self.assertTrue(client.is_closed)

    async def test_health_expects_deployed_builders_and_catalog_models(self) -> None:
        def serve(request: Request) -> Response:
            return Response(
                200,
                json={
                    "/builders/heatmap": "heatmap",
                    "/prostate-classifier-1": "prostate-classifier-1",
                },
            )

        async with AsyncClient(transport=MockTransport(serve)) as serve_client:
            with patch(
                "model_service.api.app_factory.httpx.AsyncClient",
                return_value=serve_client,
            ):
                app = build_api_process(self.settings(BUILDER_IDS='["heatmap"]'))
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://model-service"
            ) as client:
                health = await client.get("/healthz/deployments")

        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.json()["missing_routes"], ["/virchow2"])


if __name__ == "__main__":
    unittest.main()
