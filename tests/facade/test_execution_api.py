"""Check the public API: catalog listings, health and safe errors."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

from httpx import (
    ASGITransport,
    AsyncClient,
    MockTransport,
    Response,
)

from model_service.builders.contracts import (
    SpatialScoresModel,
    TileEmbeddingModel,
)
from model_service.facade.api import create_app
from model_service.facade.catalog import ModelCatalog
from model_service.facade.health import DeploymentHealth


if TYPE_CHECKING:
    from fastapi import FastAPI


PROSTATE = SpatialScoresModel(
    id="prostate-classifier-1",
    tile_size=512,
    output_tile_size=1,
    n_channels=1,
    mpp=0.468,
)
VIRCHOW = TileEmbeddingModel(id="virchow2", tile_size=224, mpp=0.5)
CATALOG = ModelCatalog([PROSTATE, VIRCHOW])
RUN = "/v1/builders/heatmap/models/prostate-classifier-1/run"


class ApiTestCase(unittest.IsolatedAsyncioTestCase):
    def client(self, app: FastAPI) -> AsyncClient:
        return AsyncClient(
            transport=ASGITransport(app=app), base_url="http://model-service"
        )

    def app(self, **kwargs: object) -> FastAPI:
        options: dict[str, object] = {"catalog": CATALOG}
        options.update(kwargs)
        return create_app(**options)  # type: ignore[arg-type]

    def assert_error(
        self,
        response: Response,
        status_code: int,
        code: str,
        *,
        retryable: bool = False,
    ) -> None:
        self.assertEqual(response.status_code, status_code, response.text)
        self.assertEqual(response.json()["detail"]["code"], code)
        self.assertEqual(response.json()["detail"]["retryable"], retryable)


class CatalogRouteTests(ApiTestCase):
    async def test_models_list_contracts_parameters_and_compatible_builders(
        self,
    ) -> None:
        async with self.client(self.app()) as client:
            listed = await client.get("/v1/models")
            single = await client.get("/v1/models/virchow2")
            missing = await client.get("/v1/models/missing")

        self.assertEqual(
            listed.json(),
            [
                {
                    "id": "prostate-classifier-1",
                    "contract": "spatial-scores",
                    "parameters": {
                        "tile_size": 512,
                        "mpp": 0.468,
                        "output_tile_size": 1,
                        "n_channels": 1,
                    },
                    "builders": ["heatmap"],
                },
                {
                    "id": "virchow2",
                    "contract": "tile-embedding",
                    "parameters": {
                        "tile_size": 224,
                        "mpp": 0.5,
                    },
                    "builders": [],
                },
            ],
        )
        self.assertEqual(single.json()["id"], "virchow2")
        self.assert_error(missing, 404, "unknown_model")

    async def test_builders_list_contracts_io_and_default_configuration(self) -> None:
        async with self.client(self.app()) as client:
            response = await client.get("/v1/builders")

        builders = {item["id"]: item for item in response.json()}
        self.assertEqual(set(builders), {"heatmap"})
        heatmap = builders["heatmap"]
        self.assertEqual(heatmap["accepts"], "spatial-scores")
        self.assertEqual(
            heatmap["inputs"]["region_of_interest"],
            {"required": False, "media_type": "image/tiff"},
        )
        self.assertTrue(heatmap["inputs"]["slide"]["required"])
        self.assertEqual(
            heatmap["outputs"],
            {"heatmap": {"kind": "shared_artifact", "media_type": "application/x-npz"}},
        )
        self.assertEqual(heatmap["configuration"]["stride_fraction"], 0.5)


class HealthAndAvailabilityTests(ApiTestCase):
    def health(self, client: AsyncClient) -> DeploymentHealth:
        return DeploymentHealth(
            client,
            origin="http://ray",
            routes=["/builders/heatmap", "/prostate-classifier-1"],
        )

    async def test_missing_routes_are_separate_from_liveness(self) -> None:
        routes = {"/builders/heatmap/": "heatmap"}
        async with (
            AsyncClient(
                transport=MockTransport(lambda _request: Response(200, json=routes))
            ) as ray,
            self.client(self.app(deployment_health=self.health(ray))) as client,
        ):
            live = await client.get("/healthz")
            deployments = await client.get("/healthz/deployments")

        self.assertEqual(live.status_code, 200)
        self.assertEqual(deployments.status_code, 503)
        self.assertEqual(
            deployments.json(),
            {
                "status": "missing_applications",
                "missing_routes": ["/prostate-classifier-1"],
            },
        )

    async def test_all_routes_present_is_ready(self) -> None:
        routes = {"/builders/heatmap": "heatmap", "/prostate-classifier-1": "p"}
        async with (
            AsyncClient(
                transport=MockTransport(lambda _request: Response(200, json=routes))
            ) as ray,
            self.client(self.app(deployment_health=self.health(ray))) as client,
        ):
            deployments = await client.get("/healthz/deployments")
        async with self.client(self.app()) as client:
            unconfigured = await client.get("/healthz/deployments")

        self.assertEqual(deployments.status_code, 200)
        self.assertEqual(deployments.json()["status"], "ready")
        self.assertEqual(unconfigured.json()["status"], "not_configured")


if __name__ == "__main__":
    unittest.main()
