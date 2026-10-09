"""Exercise workflow HTTP seams with native files and fake external model services."""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from fastapi import FastAPI, Request, Response

from execution_scheduler.persistence.memory import InMemoryExecutionRepository
from execution_scheduler.scheduler import ExecutionScheduler
from integrations.standalone import StandaloneProviderAdapter
from model_service.api import create_execution_app
from model_service.synchronous import SynchronousWorkflowService
from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.dispatch import HTTPWorkflowDispatch
from workflow_execution.http import WorkflowRunRequest, run_workflow_http


def write_slide(path: Path, pixels: np.ndarray) -> None:
    """Write a tiled Aperio fixture with known 0.5-micrometer resolution.

    Args:
        path: Local TIFF destination in temporary test storage.
        pixels: HWC uint8 slide or mask fixture pixels.

    Returns:
        None: The fixture is prepared or the specified behavior is verified.
    """
    import pyvips

    height, width = pixels.shape[:2]
    image = pyvips.Image.new_from_array(pixels)
    image.set_type(
        pyvips.GValue.gstr_type,
        "image-description",
        f"Aperio Image Library v12.0.0\n{width}x{height} |AppMag = 20|MPP = 0.5",
    )
    image.tiffsave(
        str(path),
        tile=True,
        tile_width=128,
        tile_height=128,
        compression="deflate",
        xres=2000,
        yres=2000,
    )


class AdditionalWorkflowTests(unittest.IsolatedAsyncioTestCase):
    """Check slide processing, artifact publication, and generic durable execution."""

    async def asyncSetUp(self) -> None:
        """Create isolated native slide and mask inputs for each HTTP scenario.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pixels = np.full((64, 96, 3), 80, dtype=np.uint8)
        write_slide(self.root / "slide.tiff", self.pixels)
        write_slide(self.root / "mask.tiff", np.full_like(self.pixels, 255))

    async def client(
        self,
        workflow_id: str,
        *,
        model_transport: httpx.AsyncBaseTransport | None = None,
        **worker_options: object,
    ) -> httpx.AsyncClient:
        """Compose execution modules around fake external Ray and raw-model services.

        Args:
            workflow_id: Built-in computation exposed through the execution HTTP routes.
            model_transport: Fake external HTTP transport, with no network access.
            **worker_options: Deployment-only read-thread and concurrency overrides.

        Returns:
            httpx.AsyncClient: Public HTTP client closed by test cleanup.
        """
        from workflows.registry import build_workflow
        from workflows.settings import WorkflowWorkerSettings

        self.catalog = WorkflowCatalog.default()
        application = WorkflowApplication(
            self.catalog.get(workflow_id),
            build_workflow(
                workflow_id,
                WorkflowWorkerSettings(**worker_options),
                model_transport=model_transport,
            ),
            catalog=self.catalog,
        )
        ray = FastAPI()

        @ray.post(f"/workflows/{workflow_id}/", response_model=None)
        async def run(body: WorkflowRunRequest):
            """Execute the simulated Ray ingress using the actual workflow HTTP contract.

            Args:
                body: Typed internal invocation with private execution context.

            Returns:
                WorkflowResult: Completed artifacts from the real workflow implementation.
            """
            return await run_workflow_http(application, body, body.execution_context)

        model_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=ray), base_url="http://ray"
        )
        self.addAsyncCleanup(model_client.aclose)
        dispatch = HTTPWorkflowDispatch(model_client, origin="http://ray")
        provider = StandaloneProviderAdapter(self.catalog)
        self.scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(),
            dispatch=dispatch,
            providers={provider.name: provider},
            known_workflows=self.catalog.identifiers,
            default_timeout=timedelta(minutes=1),
            maximum_timeout=timedelta(hours=1),
            poll_interval=0.01,
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_execution_app(
                    self.scheduler,
                    workflow_runner=SynchronousWorkflowService(self.catalog, dispatch),
                )
            ),
            base_url="http://scheduler",
        )
        self.addAsyncCleanup(client.aclose)
        return client

    def invocation(
        self,
        workflow_id: str,
        *,
        output_path: str | None = None,
        scheduled: bool = False,
    ) -> dict[str, object]:
        """Package shared inputs without exposing processing configuration.

        Args:
            workflow_id: Known built-in workflow being exercised by this fixture.
            output_path: Optional relative output destination.
            scheduled: Omit the deadline when admission owns the execution budget.

        Returns:
            dict[str, object]: JSON-compatible public workflow invocation.
        """
        self.catalog.get(workflow_id)
        body: dict[str, object] = {
            "inputs": {
                "slide": {"kind": "shared_path", "path": "slide.tiff"},
                "region_of_interest": {"kind": "shared_path", "path": "mask.tiff"},
            },
            "input_access": {
                "kind": "shared_storage",
                "state": {"root": str(self.root)},
            },
        }
        if output_path is not None:
            body["output_path"] = output_path
        if not scheduled:
            body["deadline_at"] = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
        return body

    async def test_episeg_publishes_float_scores_without_heatmap_quantization(
        self,
    ) -> None:
        """Verify episeg publishes float scores without heatmap quantization.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import pyvips

        model = AsyncMock()
        model.get_config.remote.return_value = {
            "tile_size": 32,
            "output_tile_size": 32,
            "n_channels": 1,
            "mpp": 0.5,
        }
        model.predict.remote.return_value = np.full((32, 32), 0.375, dtype=np.float32)
        with patch("ray.serve.get_app_handle", return_value=model):
            client = await self.client("episeg-segmentation")
            response = await client.post(
                "/v1/workflows/episeg-segmentation/run",
                json=self.invocation("episeg-segmentation"),
            )
        self.assertEqual(response.status_code, 200, response.text)
        artifact = response.json()["outputs"]["segmentation"]
        self.assertEqual(artifact["media_type"], "image/tiff")
        image = pyvips.Image.new_from_file(artifact["path"])
        self.assertEqual(
            (image.width, image.height, image.bands, image.format), (96, 64, 1, "float")
        )
        np.testing.assert_allclose(
            np.frombuffer(image.write_to_memory(), dtype=np.float32), 0.375
        )
        self.assertEqual(list(self.root.glob(".*.tiff")), [])

    async def test_gigapath_returns_features_with_original_tile_positions(self) -> None:
        """Verify gigapath returns features with original tile positions.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import h5py
        import lz4.frame

        pixels = np.full((224, 448, 3), 80, dtype=np.uint8)
        write_slide(self.root / "slide.tiff", pixels)
        write_slide(self.root / "mask.tiff", np.full_like(pixels, 255))

        async def model(request: httpx.Request) -> httpx.Response:
            """Simulate raw inference behavior for this HTTP scenario.

            Args:
                request: Fixture raw-model HTTP request.

            Returns:
                Response: Fixture response, or cancellation in the blocking scenario.
            """
            self.assertEqual(request.url.path, "/prov-gigapath/")
            self.assertEqual(len(lz4.frame.decompress(request.content)), 224 * 224 * 3)
            self.assertEqual(request.headers["x-output-dtype"], "float32")
            return httpx.Response(
                200,
                headers={"x-output-shape": "(3,)"},
                content=lz4.frame.compress(
                    np.array([1.5, -2.0, 0.25], dtype=np.float32).tobytes()
                ),
            )

        client = await self.client(
            "gigapath-embeddings", model_transport=httpx.MockTransport(model)
        )
        response = await client.post(
            "/v1/workflows/gigapath-embeddings/run",
            json=self.invocation("gigapath-embeddings"),
        )
        self.assertEqual(response.status_code, 200, response.text)
        with h5py.File(
            response.json()["outputs"]["embeddings"]["path"], "r"
        ) as artifact:
            np.testing.assert_array_equal(
                artifact["coordinates"][:], [[0, 0], [224, 0]]
            )
            np.testing.assert_array_equal(
                artifact["features"][:], [[1.5, -2.0, 0.25]] * 2
            )
            self.assertEqual(artifact.attrs["coordinate_space"], "slide_level_0")
            self.assertEqual(artifact.attrs["model_id"], "prov-gigapath")
            self.assertEqual(artifact.attrs["mpp_x"], 0.5)

    async def test_virchow_preserves_both_summaries_per_selected_tile(self) -> None:
        """Verify virchow preserves both summaries per selected tile.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import h5py
        import lz4.frame

        async def model(request: httpx.Request) -> httpx.Response:
            """Simulate raw inference behavior for this HTTP scenario.

            Args:
                request: Fixture raw-model HTTP request.

            Returns:
                Response: Fixture response, or cancellation in the blocking scenario.
            """
            self.assertEqual(request.url.path, "/virchow2/")
            self.assertEqual(request.headers["x-pool-tokens"], "true")
            return httpx.Response(
                200,
                headers={"x-output-shape": "(2, 3)"},
                content=lz4.frame.compress(
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32).tobytes()
                ),
            )

        client = await self.client(
            "virchow2-embeddings", model_transport=httpx.MockTransport(model)
        )
        response = await client.post(
            "/v1/workflows/virchow2-embeddings/run",
            json=self.invocation("virchow2-embeddings"),
        )
        self.assertEqual(response.status_code, 200, response.text)
        with h5py.File(
            response.json()["outputs"]["embeddings"]["path"], "r"
        ) as artifact:
            np.testing.assert_array_equal(
                artifact["features"][:], [[[1, 2, 3], [4, 5, 6]]]
            )
            np.testing.assert_array_equal(artifact["coordinates"][:], [[0, 0]])
            self.assertEqual(
                list(artifact.attrs["summary_names"]), ["global", "local_mean"]
            )

    async def test_all_new_workflows_use_durable_admission_and_attempt_outputs(
        self,
    ) -> None:
        """Verify all new workflows use durable admission and attempt outputs.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import h5py
        import lz4.frame

        for workflow_id in (
            "episeg-segmentation",
            "gigapath-embeddings",
            "virchow2-embeddings",
        ):
            with self.subTest(workflow_id=workflow_id):
                shape = (2, 3) if workflow_id == "virchow2-embeddings" else (3,)
                model_transport = httpx.MockTransport(
                    lambda _request, shape=shape: httpx.Response(
                        200,
                        headers={"x-output-shape": str(shape)},
                        content=lz4.frame.compress(
                            np.ones(shape, dtype=np.float32).tobytes()
                        ),
                    )
                )
                ray_model = AsyncMock()
                ray_model.get_config.remote.return_value = {
                    "tile_size": 32,
                    "output_tile_size": 32,
                    "n_channels": 1,
                    "mpp": 0.5,
                }
                ray_model.predict.remote.return_value = np.full((32, 32), 0.375)
                with patch("ray.serve.get_app_handle", return_value=ray_model):
                    client = await self.client(
                        workflow_id, model_transport=model_transport
                    )
                    submission = {
                        "provider": "standalone",
                        "idempotency_key": workflow_id,
                        "body": self.invocation(workflow_id, scheduled=True),
                    }
                    response = await client.post(
                        f"/v1/workflows/{workflow_id}/schedule", json=submission
                    )
                    self.assertEqual(response.status_code, 202, response.text)
                    execution_id = response.json()["execution"]["id"]
                    replay = await client.post(
                        f"/v1/workflows/{workflow_id}/schedule", json=submission
                    )
                    self.assertEqual(replay.json()["execution"]["id"], execution_id)
                    async with self.scheduler.running():
                        async with asyncio.timeout(5):
                            while True:
                                execution = (
                                    await client.get(f"/v1/executions/{execution_id}")
                                ).json()
                                if execution["status"] in {"succeeded", "failed"}:
                                    break
                                await asyncio.sleep(0.01)
                    self.assertEqual(execution["status"], "succeeded", execution)
                    output = next(iter(execution["result"]["outputs"].values()))
                    self.assertIn(".workflow-attempts", output["path"])
                    self.assertTrue(Path(output["path"]).is_file())
                    if workflow_id != "episeg-segmentation":
                        with h5py.File(output["path"], "r") as artifact:
                            self.assertEqual(artifact["features"].shape, (1, *shape))

    async def test_empty_region_produces_empty_features_without_model_calls(
        self,
    ) -> None:
        """Verify empty region produces empty features without model calls.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import h5py

        write_slide(self.root / "mask.tiff", np.zeros_like(self.pixels))

        def reject_model_call(_request):
            """Fail if an empty region attempts any raw inference.

            Args:
                _request: Unexpected raw-model request which fails this test.

            Returns:
                None: The fixture is prepared or the specified behavior is verified.
            """
            self.fail("an empty region must not call the model")

        client = await self.client(
            "virchow2-embeddings",
            model_transport=httpx.MockTransport(reject_model_call),
        )
        response = await client.post(
            "/v1/workflows/virchow2-embeddings/run",
            json=self.invocation("virchow2-embeddings"),
        )
        self.assertEqual(response.status_code, 200, response.text)
        with h5py.File(
            response.json()["outputs"]["embeddings"]["path"], "r"
        ) as artifact:
            self.assertEqual(artifact["features"].shape, (0, 2, 0))
            self.assertEqual(artifact["coordinates"].shape, (0, 2))

    async def test_invalid_model_responses_leave_no_partial_artifacts(self) -> None:
        """Verify invalid model responses leave no partial artifacts.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import lz4.frame

        for headers, values in (
            ({"x-output-shape": "(1, 3)"}, [1, 2, 3]),
            ({"x-output-shape": "(3,)"}, [1, 2]),
            ({"x-output-shape": "(3,)"}, [1, float("nan"), 3]),
            ({}, [1, 2, 3]),
        ):
            with self.subTest(headers=headers, values=values):
                client = await self.client(
                    "gigapath-embeddings",
                    model_transport=httpx.MockTransport(
                        lambda _request, headers=headers, values=values: httpx.Response(
                            200,
                            headers=headers,
                            content=lz4.frame.compress(
                                np.array(values, dtype=np.float32).tobytes()
                            ),
                        )
                    ),
                )
                response = await client.post(
                    "/v1/workflows/gigapath-embeddings/run",
                    json=self.invocation("gigapath-embeddings"),
                )
                self.assertEqual(response.status_code, 422, response.text)
                self.assertFalse(response.json()["detail"]["retryable"])
                self.assertEqual(list(self.root.glob("*.h5")), [])
                self.assertEqual(list(self.root.glob(".*.h5")), [])

    async def test_model_outage_is_retryable_and_does_not_publish(self) -> None:
        """Verify model outage is retryable and does not publish.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        client = await self.client(
            "gigapath-embeddings",
            model_transport=httpx.MockTransport(lambda _request: httpx.Response(503)),
        )
        response = await client.post(
            "/v1/workflows/gigapath-embeddings/run",
            json=self.invocation("gigapath-embeddings"),
        )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertTrue(response.json()["detail"]["retryable"])
        self.assertEqual(list(self.root.glob("*.h5")), [])
        self.assertEqual(list(self.root.glob(".*.h5")), [])

    async def test_scheduled_cancellation_joins_model_requests_and_removes_partial_output(
        self,
    ) -> None:
        """Verify scheduled cancellation joins model requests and removes partial output.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        entered = asyncio.Event()
        stopped = asyncio.Event()

        async def model(_request):
            """Simulate raw inference behavior for this HTTP scenario.

            Args:
                _request: Fixture raw-model HTTP request.

            Returns:
                Response: Fixture response, or cancellation in the blocking scenario.
            """
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        client = await self.client(
            "gigapath-embeddings", model_transport=httpx.MockTransport(model)
        )
        response = await client.post(
            "/v1/workflows/gigapath-embeddings/schedule",
            json={
                "provider": "standalone",
                "idempotency_key": "cancel",
                "body": self.invocation("gigapath-embeddings", scheduled=True),
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        execution_id = response.json()["execution"]["id"]
        async with self.scheduler.running():
            await asyncio.wait_for(entered.wait(), 5)
            cancelled = await client.post(f"/v1/executions/{execution_id}/cancel")
            self.assertEqual(cancelled.json()["status"], "cancelled", cancelled.text)
            await asyncio.wait_for(stopped.wait(), 5)
        self.assertEqual(list(self.root.rglob("*.h5")), [])
        self.assertEqual(list(self.root.rglob(".*.h5")), [])

    async def test_region_selection_keeps_only_intersecting_tile_positions(
        self,
    ) -> None:
        """Verify region selection keeps only intersecting tile positions.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import h5py
        import lz4.frame

        pixels = np.full((224, 448, 3), 80, dtype=np.uint8)
        mask = np.zeros_like(pixels)
        mask[:, 224:] = 255
        write_slide(self.root / "slide.tiff", pixels)
        write_slide(self.root / "mask.tiff", mask)
        client = await self.client(
            "gigapath-embeddings",
            model_transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    headers={"x-output-shape": "(3,)"},
                    content=lz4.frame.compress(np.ones(3, dtype=np.float32).tobytes()),
                )
            ),
        )
        response = await client.post(
            "/v1/workflows/gigapath-embeddings/run",
            json=self.invocation("gigapath-embeddings"),
        )
        self.assertEqual(response.status_code, 200, response.text)
        with h5py.File(
            response.json()["outputs"]["embeddings"]["path"], "r"
        ) as artifact:
            np.testing.assert_array_equal(artifact["coordinates"][:], [[224, 0]])

    async def test_feature_dimension_drift_preserves_previous_completed_output(
        self,
    ) -> None:
        """Verify feature dimension drift preserves previous completed output.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import lz4.frame

        pixels = np.full((224, 448, 3), 80, dtype=np.uint8)
        write_slide(self.root / "slide.tiff", pixels)
        write_slide(self.root / "mask.tiff", np.full_like(pixels, 255))
        (self.root / "embeddings.h5").write_bytes(b"previous complete output")
        sizes = iter((3, 4))

        def model(_request):
            """Simulate raw inference behavior for this HTTP scenario.

            Args:
                _request: Fixture raw-model HTTP request.

            Returns:
                Response: Fixture response, or cancellation in the blocking scenario.
            """
            size = next(sizes)
            return httpx.Response(
                200,
                headers={"x-output-shape": str((size,))},
                content=lz4.frame.compress(np.ones(size, dtype=np.float32).tobytes()),
            )

        client = await self.client(
            "gigapath-embeddings", model_transport=httpx.MockTransport(model)
        )
        response = await client.post(
            "/v1/workflows/gigapath-embeddings/run",
            json=self.invocation("gigapath-embeddings"),
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(
            (self.root / "embeddings.h5").read_bytes(), b"previous complete output"
        )
        self.assertEqual(list(self.root.glob(".*.h5")), [])

    async def test_output_cannot_replace_slide_input(self) -> None:
        """Verify output cannot replace slide input.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        original = (self.root / "slide.tiff").read_bytes()
        client = await self.client("gigapath-embeddings")
        response = await client.post(
            "/v1/workflows/gigapath-embeddings/run",
            json=self.invocation("gigapath-embeddings", output_path="slide.tiff"),
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual((self.root / "slide.tiff").read_bytes(), original)

    async def test_raw_model_ingress_does_not_require_following_a_redirect(
        self,
    ) -> None:
        """Verify raw model ingress does not require following a redirect.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import lz4.frame

        model = FastAPI()

        @model.post("/")
        async def predict(request: Request) -> Response:
            """Serve the unchanged slash-terminated raw RGB/LZ4 inference contract.

            Args:
                request: Compressed RGB tile request to the mounted model ingress.

            Returns:
                Response: Fixture response, or cancellation in the blocking scenario.
            """
            self.assertEqual(
                len(lz4.frame.decompress(await request.body())), 224 * 224 * 3
            )
            return Response(
                content=lz4.frame.compress(np.ones(3, dtype=np.float32).tobytes()),
                headers={"x-output-shape": "(3,)"},
            )

        ray = FastAPI()
        ray.mount("/prov-gigapath", model)
        client = await self.client(
            "gigapath-embeddings", model_transport=httpx.ASGITransport(app=ray)
        )
        response = await client.post(
            "/v1/workflows/gigapath-embeddings/run",
            json=self.invocation("gigapath-embeddings"),
        )
        self.assertEqual(response.status_code, 200, response.text)

    async def test_downsampled_mask_does_not_drop_selected_edge_pixels(self) -> None:
        """Verify downsampled mask does not drop selected edge pixels.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import h5py
        import lz4.frame

        pixels = np.full((224, 448, 3), 80, dtype=np.uint8)
        write_slide(self.root / "slide.tiff", pixels)
        write_slide(
            self.root / "mask.tiff",
            np.array([[[0, 0, 0], [0, 0, 0], [255, 255, 255]]], dtype=np.uint8),
        )
        client = await self.client(
            "gigapath-embeddings",
            model_transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    headers={"x-output-shape": "(3,)"},
                    content=lz4.frame.compress(np.ones(3, dtype=np.float32).tobytes()),
                )
            ),
        )
        response = await client.post(
            "/v1/workflows/gigapath-embeddings/run",
            json=self.invocation("gigapath-embeddings"),
        )
        self.assertEqual(response.status_code, 200, response.text)
        with h5py.File(
            response.json()["outputs"]["embeddings"]["path"], "r"
        ) as artifact:
            np.testing.assert_array_equal(artifact["coordinates"][:], [[224, 0]])

    async def test_episeg_pads_a_slide_smaller_than_its_model_tile(self) -> None:
        """Verify episeg pads a slide smaller than its model tile.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import pyvips

        model = AsyncMock()
        model.get_config.remote.return_value = {
            "tile_size": 1024,
            "output_tile_size": 1024,
            "n_channels": 1,
            "mpp": 0.5,
        }
        model.predict.remote.return_value = np.full(
            (1024, 1024), 0.375, dtype=np.float32
        )
        with patch("ray.serve.get_app_handle", return_value=model):
            client = await self.client("episeg-segmentation")
            response = await client.post(
                "/v1/workflows/episeg-segmentation/run",
                json=self.invocation("episeg-segmentation"),
            )
        self.assertEqual(response.status_code, 200, response.text)
        image = pyvips.Image.new_from_file(
            response.json()["outputs"]["segmentation"]["path"]
        )
        np.testing.assert_allclose(
            np.frombuffer(image.write_to_memory(), dtype=np.float32), 0.375
        )

    async def test_plain_tiff_mask_does_not_need_slide_mpp_metadata(self) -> None:
        """Verify plain tiff mask does not need slide mpp metadata.

        Returns:
            None: The fixture is prepared or the specified behavior is verified.
        """
        import lz4.frame
        import tifffile

        tifffile.imwrite(
            self.root / "mask.tiff",
            np.full_like(self.pixels, 255),
            tile=(128, 128),
            resolutionunit="NONE",
            metadata=None,
        )
        for workflow_id in ("episeg-segmentation", "gigapath-embeddings"):
            with self.subTest(workflow_id=workflow_id):
                model = AsyncMock()
                model.get_config.remote.return_value = {
                    "tile_size": 32,
                    "output_tile_size": 32,
                    "n_channels": 1,
                    "mpp": 0.5,
                }
                model.predict.remote.return_value = np.full((32, 32), 0.375)
                with patch("ray.serve.get_app_handle", return_value=model):
                    client = await self.client(
                        workflow_id,
                        model_transport=httpx.MockTransport(
                            lambda _request: httpx.Response(
                                200,
                                headers={"x-output-shape": "(3,)"},
                                content=lz4.frame.compress(
                                    np.ones(3, dtype=np.float32).tobytes()
                                ),
                            )
                        ),
                    )
                    response = await client.post(
                        f"/v1/workflows/{workflow_id}/run",
                        json=self.invocation(workflow_id),
                    )
                self.assertEqual(response.status_code, 200, response.text)
