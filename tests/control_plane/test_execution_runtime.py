"""Check production wiring: catalog loading, lifecycle and per-model raw origins."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response

from model_service.control_plane.app_factory import build_control_plane
from model_service.control_plane.catalog_files import CatalogError
from model_service.control_plane.scheduler.settings import SchedulerSettings
from tests.support.memory_repository import InMemoryExecutionRepository
from tests.support.model_documents import served_document


DEPLOYED = {"prostate-classifier-1": served_document(), "virchow2": served_document()}


class RuntimeTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.catalog_dir = self.directory / "catalog-mount"
        self.catalog_dir.mkdir()
        self.catalog_path = self.catalog_dir / "catalog.json"
        self.write_served(DEPLOYED)

    def write_served(self, documents: dict[str, object]) -> None:
        (self.catalog_dir / "catalog.json").write_text(
            json.dumps({"served-models": documents}), encoding="utf-8"
        )

    def settings(self, **environment: str) -> SchedulerSettings:
        values = {
            "DATABASE_URL": "postgresql://unused",
            "PROVIDER_CONTEXT_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
            "RAY_SERVE_ORIGIN": "http://ray-serve",
            "MODEL_CATALOG_DIR": str(self.catalog_dir),
            "SHARED_STORAGE_ROOTS": json.dumps([str(self.directory)]),
            **environment,
        }
        with patch.dict(os.environ, values, clear=True):
            return SchedulerSettings()


class ExecutionRuntimeTests(RuntimeTestCase):
    async def test_construction_loads_catalog_without_contacting_the_database(
        self,
    ) -> None:
        app = build_control_plane(self.settings())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://model-service"
        ) as client:
            models = await client.get("/v1/models")

        self.assertEqual(
            [model["id"] for model in models.json()],
            ["prostate-classifier-1", "virchow2"],
        )

    def test_an_unreadable_or_invalid_catalog_prevents_startup(self) -> None:
        self.catalog_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(CatalogError):
            build_control_plane(self.settings())
        self.write_served({"small": served_document(tile_size=0)})
        with self.assertRaises(CatalogError):
            build_control_plane(self.settings())
        self.catalog_path.unlink()
        with self.assertRaises(CatalogError):
            build_control_plane(self.settings())

    async def test_lifespan_initializes_repository_and_closes_client(self) -> None:
        repository = InMemoryExecutionRepository()
        initialized = AsyncMock()
        maintained = asyncio.Event()
        tasks_before = asyncio.all_tasks()
        client = AsyncClient(transport=MockTransport(lambda _request: Response(503)))
        with (
            patch.object(repository, "initialize", initialized, create=True),
            patch.object(repository, "maintain", side_effect=maintained.set),
            patch(
                "model_service.control_plane.app_factory.PostgresExecutionRepository",
                return_value=repository,
            ),
            patch(
                "model_service.control_plane.app_factory.httpx.AsyncClient",
                return_value=client,
            ),
        ):
            app = build_control_plane(self.settings())
            async with app.router.lifespan_context(app):
                initialized.assert_awaited_once()
                await asyncio.wait_for(maintained.wait(), timeout=1)
                self.assertFalse(client.is_closed)
            self.assertTrue(client.is_closed)
            self.assertEqual(asyncio.all_tasks(), tasks_before)

    async def test_lifespan_closes_client_when_initialization_fails(self) -> None:
        repository = InMemoryExecutionRepository()
        initialized = AsyncMock(side_effect=RuntimeError("database unavailable"))
        client = AsyncClient(transport=MockTransport(lambda _request: Response(503)))
        with (
            patch.object(repository, "initialize", initialized, create=True),
            patch(
                "model_service.control_plane.app_factory.PostgresExecutionRepository",
                return_value=repository,
            ),
            patch(
                "model_service.control_plane.app_factory.httpx.AsyncClient",
                return_value=client,
            ),
        ):
            app = build_control_plane(self.settings())
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                async with app.router.lifespan_context(app):
                    self.fail("startup unexpectedly succeeded")
            self.assertTrue(client.is_closed)

    async def test_a_catalog_change_reaches_the_running_control_plane(self) -> None:
        async def until(client: AsyncClient, path: str, done: object) -> Response:
            for _ in range(200):
                response = await client.get(path)
                if done(response):  # type: ignore[operator]
                    return response
                await asyncio.sleep(0.01)
            self.fail(f"{path} never changed: {response.json()}")

        def ids(response: Response) -> list[str]:
            return [model["id"] for model in response.json()]

        def serve(_request: Request) -> Response:
            return Response(200, json={"/prostate-classifier-1": "p", "/virchow2": "v"})

        repository = InMemoryExecutionRepository()
        serve_client = AsyncClient(transport=MockTransport(serve))
        with (
            patch.object(repository, "initialize", AsyncMock(), create=True),
            patch(
                "model_service.control_plane.app_factory.PostgresExecutionRepository",
                return_value=repository,
            ),
            patch(
                "model_service.control_plane.app_factory.httpx.AsyncClient",
                return_value=serve_client,
            ),
        ):
            app = build_control_plane(self.settings(CATALOG_RELOAD_SECONDS="0.01"))
            async with (
                app.router.lifespan_context(app),
                AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://model-service"
                ) as client,
            ):
                self.write_served({**DEPLOYED, "small": served_document()})
                added = await until(client, "/v1/models", lambda r: "small" in ids(r))
                deployments = await client.get("/healthz/deployments")
                self.write_served({**DEPLOYED, "small": served_document(tile_size=0)})
                failed = await until(
                    client, "/healthz/catalog", lambda r: r.status_code == 503
                )
                kept = await client.get("/v1/models")

        self.assertEqual(
            sorted(ids(added)), ["prostate-classifier-1", "small", "virchow2"]
        )
        self.assertEqual(deployments.json()["missing_routes"], ["/small"])
        self.assertEqual(failed.json()["status"], "reload_failed")
        self.assertIn(
            "catalog/served-models/small.yaml:input.tile_size", failed.json()["error"]
        )
        self.assertEqual(ids(kept), ids(added))

    async def test_health_expects_every_deployed_served_model(self) -> None:
        def serve(request: Request) -> Response:
            return Response(
                200, json={"/prostate-classifier-1": "prostate-classifier-1"}
            )

        async with AsyncClient(transport=MockTransport(serve)) as serve_client:
            with patch(
                "model_service.control_plane.app_factory.httpx.AsyncClient",
                return_value=serve_client,
            ):
                app = build_control_plane(self.settings())
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://model-service"
            ) as client:
                health = await client.get("/healthz/deployments")

        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.json()["missing_routes"], ["/virchow2"])


if __name__ == "__main__":
    unittest.main()
