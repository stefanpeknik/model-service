"""End to end for tile embeddings: API, scheduler, dispatch, builder, fake model."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import numpy as np
from fastapi import FastAPI

from model_service.analysis_runner.settings import ReplicaSettings
from model_service.api.app_factory import create_app, model_resolver
from model_service.api.model_catalog import ModelCatalog
from model_service.api.scheduler.builder_dispatch import HTTPBuilderDispatch
from model_service.api.scheduler.execution_scheduler import ExecutionScheduler
from model_service.api.synchronous_runs import SynchronousBuilderService
from model_service.builders.implementations.embeddings.builder import EmbeddingBuilder
from model_service.builders.implementations.embeddings.spec import (
    EmbeddingConfiguration,
)
from model_service.builders.registry import get_builder
from model_service.builders.replica.http_handler import run_http
from model_service.builders.replica.runner import BuilderApplication
from model_service.contracts.builder_http import RunRequest
from model_service.contracts.model_declarations import TileEmbeddingModel
from model_service.providers.standalone.adapter import StandaloneProviderAdapter
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from model_service.testing import fake_slide
from tests.support.builder_contract import BuilderContractTests
from tests.support.memory_repository import InMemoryExecutionRepository


if TYPE_CHECKING:
    from collections.abc import Callable


try:
    import h5py
    import pyvips
except (ImportError, OSError):
    HAS_NATIVE = False
else:
    HAS_NATIVE = bool(pyvips.Image)

TILE = 32


class _Method:
    def __init__(self, function: Callable[..., object]) -> None:
        self._function = function

    async def remote(self, *args: object) -> object:
        return self._function(*args)


class FakeEmbeddingDeployment:
    """Stands in for a deployment handle: the feature is the tile's mean value."""

    def __init__(self, summaries: tuple[str, ...]) -> None:
        self.summaries = summaries
        self.tile_size = TILE
        self.calls = 0
        self.failure: Exception | None = None
        self.dimension = 4
        self.get_config = _Method(
            lambda: {
                "tile_size": self.tile_size,
                "mpp": 0.5,
                "summaries": list(self.summaries),
            }
        )
        self.embed = _Method(self._embed)

    def _embed(self, tile: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        assert tile.shape == (3, TILE, TILE) and tile.dtype == np.uint8
        shape = (
            (self.dimension,)
            if len(self.summaries) == 1
            else (len(self.summaries), self.dimension)
        )
        return np.full(shape, float(tile.mean()), dtype=np.float32)


@unittest.skipUnless(HAS_NATIVE, "libvips or h5py is unavailable")
class WithoutTissueFilter(EmbeddingConfiguration):
    """Defaults with the tissue filter off.

    Tissue detection needs a realistic slide; see slides/test_tissue.py.
    """

    skip_background: bool = False


# These tests run through the public API, which takes no configuration.
SPEC = replace(get_builder("tile-embeddings"), configuration=WithoutTissueFilter)


class EmbeddingContractTests(BuilderContractTests, unittest.IsolatedAsyncioTestCase):
    builder_id = "tile-embeddings"
    configuration: ClassVar[dict[str, Any]] = {"skip_background": False}
    model = TileEmbeddingModel(
        id="encoder", tile_size=TILE, mpp=0.5, summaries=("global", "mean")
    )


@unittest.skipUnless(HAS_NATIVE, "libvips, h5py or lz4 is unavailable")
class EmbeddingBuilderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # Level 0 at 0.25 mpp; the model wants 0.5 mpp, i.e. level 1 (64x48).
        pixels = np.zeros((96, 128, 3), dtype=np.uint8)
        pixels[:, 64:] = 200
        fake_slide(self.root / "slide.tiff", pixels, mpp=0.25, pyramid=True, tile=16)
        mask = np.zeros((96, 128, 3), dtype=np.uint8)
        mask[:, 64:] = 255
        fake_slide(self.root / "mask.tiff", mask, mpp=0.25)

    async def client(self, summaries: tuple[str, ...]) -> httpx.AsyncClient:
        self.model = FakeEmbeddingDeployment(summaries)
        catalog = ModelCatalog(
            [
                TileEmbeddingModel(
                    id="encoder", tile_size=TILE, mpp=0.5, summaries=summaries
                )
            ]
        )
        storage = SharedStorageAccessFactory([self.root])
        settings = ReplicaSettings(num_threads=1, max_concurrent_tasks=3)
        application = BuilderApplication(
            SPEC,
            EmbeddingBuilder(settings),
            {"shared_storage": storage},
            model_handles=lambda _model_id: self.model,  # type: ignore[arg-type, return-value]
        )
        ray = FastAPI()

        @ray.post("/builders/tile-embeddings/", response_model=None)
        async def run(body: RunRequest):  # type: ignore[no-untyped-def]
            return await run_http(application, body)

        ray_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=ray), base_url="http://ray"
        )
        self.addAsyncCleanup(ray_client.aclose)
        dispatch = HTTPBuilderDispatch(ray_client, origin="http://ray")
        self.scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(),
            dispatch=dispatch,
            providers={"standalone": StandaloneProviderAdapter(storage)},
            resolve_model=model_resolver(catalog),
            default_timeout=timedelta(minutes=1),
            maximum_timeout=timedelta(hours=1),
            poll_interval=0.01,
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_app(
                    self.scheduler,
                    catalog=catalog,
                    builder_runner=SynchronousBuilderService(catalog, dispatch),
                )
            ),
            base_url="http://api",
        )
        self.addAsyncCleanup(client.aclose)
        return client

    def body(self, *, mask: bool = True, scheduled: bool = False) -> dict[str, object]:
        inputs: dict[str, object] = {
            "slide": {"kind": "shared_path", "path": "slide.tiff"}
        }
        if mask:
            inputs["region_of_interest"] = {"kind": "shared_path", "path": "mask.tiff"}
        body: dict[str, object] = {
            "inputs": inputs,
            "input_access": {
                "kind": "shared_storage",
                "state": {"root": str(self.root)},
            },
            "output_path": "out/features.h5",
        }
        if not scheduled:
            body["deadline_at"] = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
        return body

    async def test_features_are_read_at_the_level_matching_model_resolution(
        self,
    ) -> None:
        client = await self.client(("tile",))

        response = await client.post(
            "/v1/builders/tile-embeddings/models/encoder/run", json=self.body()
        )

        self.assertEqual(response.status_code, 200, response.text)
        path = response.json()["outputs"]["embeddings"]["path"]
        with h5py.File(path) as output:
            self.assertEqual(output.attrs["slide_level"], 1)
            self.assertEqual(output.attrs["slide_width"], 64)
            self.assertEqual(output.attrs["model_mpp"], 0.5)
            coordinates = output["coordinates"][:].tolist()
            features = output["features"][:]
        # Level 1 is 64x48: a 2x2 grid of 32-pixel tiles (rows padded); the mask
        # selects only the right column.
        self.assertEqual(coordinates, [[32, 0], [32, 32]])
        self.assertEqual(features.shape, (2, 4))
        self.assertTrue(np.all(features[:, 0] > 0))
        self.assertEqual(self.model.calls, 2)

    async def test_without_a_mask_every_tile_is_embedded(self) -> None:
        client = await self.client(("tile",))

        response = await client.post(
            "/v1/builders/tile-embeddings/models/encoder/run",
            json=self.body(mask=False),
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.model.calls, 4)

    async def test_rows_follow_grid_order_when_later_tiles_finish_first(
        self,
    ) -> None:
        client = await self.client(("tile",))
        embed = self.model.embed
        delays = iter([0.03, 0.02, 0.01, 0.0])

        class SlowFirst:
            async def remote(self, tile: np.ndarray) -> object:
                await asyncio.sleep(next(delays))
                return await embed.remote(tile)

        self.model.embed = SlowFirst()  # type: ignore[assignment]

        response = await client.post(
            "/v1/builders/tile-embeddings/models/encoder/run",
            json=self.body(mask=False),
        )

        self.assertEqual(response.status_code, 200, response.text)
        with h5py.File(response.json()["outputs"]["embeddings"]["path"]) as output:
            coordinates = output["coordinates"][:].tolist()
            features = output["features"][:, 0].tolist()
        self.assertEqual(coordinates, [[0, 0], [32, 0], [0, 32], [32, 32]])
        # Each tile's mean: the left half is black, the right 200, and the
        # bottom tiles' lower half is padding, read as white.
        self.assertEqual(features, [0.0, 200.0, 127.5, 227.5])

    async def test_a_caller_sets_the_builders_settings_per_run(self) -> None:
        client = await self.client(("tile",))

        response = await client.post(
            "/v1/builders/tile-embeddings/models/encoder/run",
            json=self.body() | {"configuration": {"stride_fraction": 0.5}},
        )

        self.assertEqual(response.status_code, 200, response.text)
        path = response.json()["outputs"]["embeddings"]["path"]
        with h5py.File(path) as output:
            self.assertEqual(output.attrs["stride"], 16)

    async def test_a_setting_the_builder_rejects_is_refused_with_its_rule(
        self,
    ) -> None:
        client = await self.client(("tile",))

        response = await client.post(
            "/v1/builders/tile-embeddings/models/encoder/run",
            json=self.body() | {"configuration": {"stride_fraction": 0}},
        )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("stride_fraction: Input should be greater than 0", response.text)
        self.assertEqual(self.model.calls, 0)

    async def test_summaries_are_stored_per_tile(self) -> None:
        client = await self.client(("global", "local_mean"))

        response = await client.post(
            "/v1/builders/tile-embeddings/models/encoder/run", json=self.body()
        )

        path = response.json()["outputs"]["embeddings"]["path"]
        with h5py.File(path) as output:
            self.assertEqual(output["features"].shape, (2, 2, 4))
            self.assertEqual(
                [str(name) for name in output.attrs["summary_names"]],
                ["global", "local_mean"],
            )

    async def test_scheduled_run_ends_at_the_requested_path(self) -> None:
        client = await self.client(("tile",))

        async with self.scheduler.running():
            admitted = await client.post(
                "/v1/builders/tile-embeddings/models/encoder/schedule",
                json=self.body(scheduled=True)
                | {"configuration": {"skip_background": False}},
                headers={"Idempotency-Key": "key-1"},
            )
            self.assertEqual(admitted.status_code, 202, admitted.text)
            execution_id = admitted.json()["execution"]["id"]
            for _ in range(500):
                view = (await client.get(f"/v1/executions/{execution_id}")).json()
                if view["status"] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.01)

        self.assertEqual(view["status"], "succeeded", view)
        self.assertEqual(view["models"], ["encoder"])
        # Provenance: every setting it ran with, not only the one sent.
        self.assertEqual(
            view["configuration"],
            {
                "stride_fraction": 1.0,
                "skip_background": False,
                "min_tissue_fraction": 0.0,
            },
        )
        path = Path(view["result"]["outputs"]["embeddings"]["path"])
        self.assertNotIn(".attempts", path.parts)
        self.assertTrue(path.is_file())

    async def test_model_failure_leaves_no_artifact(self) -> None:
        client = await self.client(("tile",))
        self.model.failure = RuntimeError("CUDA out of memory")

        response = await client.post(
            "/v1/builders/tile-embeddings/models/encoder/run", json=self.body()
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "model_failed")
        self.assertTrue(response.json()["detail"]["retryable"])
        self.assertFalse((self.root / "out" / "features.h5").exists())

    async def test_invalid_features_are_a_permanent_failure(self) -> None:
        client = await self.client(("tile",))
        self.model.dimension = 0

        response = await client.post(
            "/v1/builders/tile-embeddings/models/encoder/run", json=self.body()
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "invalid_model_output")
        self.assertFalse((self.root / "out" / "features.h5").exists())

    async def test_running_model_must_match_its_declaration(self) -> None:
        client = await self.client(("tile",))
        self.model.tile_size = 64

        response = await client.post(
            "/v1/builders/tile-embeddings/models/encoder/run", json=self.body()
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "model_contract_mismatch")
        self.assertEqual(self.model.calls, 0)

    async def test_heatmap_model_is_rejected_for_embeddings(self) -> None:
        client = await self.client(("tile",))
        response = await client.post(
            "/v1/builders/tile-embeddings/models/unknown/run", json=self.body()
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "unknown_model")
