"""The control plane's wiring: catalog loading, lifecycle, catalog reloads and route health."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response

from model_service.control_plane.app_factory import build_control_plane
from model_service.control_plane.catalog_files import CatalogError
from model_service.control_plane.scheduler.settings import SchedulerSettings
from tests.support.memory_repository import MemoryStorage
from tests.support.model_documents import model_document


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from fastapi import FastAPI

    from model_service.control_plane.scheduler.store.storage import StorageUnit


DEPLOYED = {"prostate-classifier-1": model_document(), "virchow2": model_document()}


class LifecycleStorage(MemoryStorage):
    """In-memory storage that records its lifecycle and can fail to initialize."""

    def __init__(self, failure: Exception | None = None) -> None:
        super().__init__()
        self.failure = failure
        self.initialized = False
        self.closed = False
        self.used = asyncio.Event()

    async def initialize(self) -> None:
        """Record the call, then fail if told to."""
        self.initialized = True
        if self.failure is not None:
            raise self.failure

    def unit(self) -> AbstractAsyncContextManager[StorageUnit]:
        """Record that something used the storage, then open a unit."""
        self.used.set()
        return super().unit()

    async def aclose(self) -> None:
        """Record the close."""
        self.closed = True


def ray_serve(answer: Callable[[Request], Response] | None = None) -> AsyncClient:
    """A client to a stand-in Ray Serve, unavailable unless told how to answer."""
    return AsyncClient(transport=MockTransport(answer or (lambda _r: Response(503))))


class RuntimeTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.catalog_dir = self.directory / "catalog-mount"
        self.catalog_dir.mkdir()
        self.catalog_path = self.catalog_dir / "catalog.json"
        self.write_model(DEPLOYED)

    def write_model(self, documents: dict[str, object]) -> None:
        (self.catalog_dir / "catalog.json").write_text(
            json.dumps({"models": documents}), encoding="utf-8"
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

    def control_plane(
        self,
        storage: MemoryStorage | None = None,
        client: AsyncClient | None = None,
        **environment: str,
    ) -> FastAPI:
        """The control plane over in-memory storage and a stand-in Ray Serve."""
        return build_control_plane(
            self.settings(**environment),
            storage=storage or MemoryStorage(),
            client=client or ray_serve(),
        )


class ExecutionRuntimeTests(RuntimeTestCase):
    async def test_construction_loads_catalog_without_contacting_the_database(
        self,
    ) -> None:
        app = self.control_plane()
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
            self.control_plane()
        self.write_model({"small": model_document(tile_size=0)})
        with self.assertRaises(CatalogError):
            self.control_plane()
        self.catalog_path.unlink()
        with self.assertRaises(CatalogError):
            self.control_plane()

    async def test_lifespan_initializes_storage_runs_the_worker_and_closes(
        self,
    ) -> None:
        storage = LifecycleStorage()
        client = ray_serve()
        tasks_before = asyncio.all_tasks()
        app = self.control_plane(storage, client)

        async with app.router.lifespan_context(app):
            self.assertTrue(storage.initialized)
            # The worker's maintenance pass reaches the storage.
            await asyncio.wait_for(storage.used.wait(), timeout=1)
            self.assertFalse(client.is_closed)

        self.assertTrue(client.is_closed)
        self.assertTrue(storage.closed)
        self.assertEqual(asyncio.all_tasks(), tasks_before)

    async def test_lifespan_closes_client_when_initialization_fails(self) -> None:
        storage = LifecycleStorage(RuntimeError("database unavailable"))
        client = ray_serve()
        app = self.control_plane(storage, client)

        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            async with app.router.lifespan_context(app):
                self.fail("startup unexpectedly succeeded")
        self.assertTrue(client.is_closed)
        self.assertTrue(storage.closed)

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

        app = self.control_plane(client=ray_serve(serve), CATALOG_RELOAD_SECONDS="0.01")
        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=ASGITransport(app=app), base_url="http://model-service"
            ) as client,
        ):
            self.write_model({**DEPLOYED, "small": model_document()})
            added = await until(client, "/v1/models", lambda r: "small" in ids(r))
            deployments = await client.get("/healthz/deployments")
            self.write_model({**DEPLOYED, "small": model_document(tile_size=0)})
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
            "catalog/models/small.yaml:input.tile_size", failed.json()["error"]
        )
        self.assertEqual(ids(kept), ids(added))

    async def test_health_expects_every_deployed_model(self) -> None:
        def serve(request: Request) -> Response:
            return Response(
                200, json={"/prostate-classifier-1": "prostate-classifier-1"}
            )

        app = self.control_plane(client=ray_serve(serve))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://model-service"
        ) as client:
            health = await client.get("/healthz/deployments")

        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.json()["missing_routes"], ["/virchow2"])


if __name__ == "__main__":
    unittest.main()
