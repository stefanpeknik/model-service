from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from fastapi import FastAPI

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
    async def asyncSetUp(self) -> None:
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
        )
        ray = FastAPI()

        @ray.post(f"/workflows/{workflow_id}/", response_model=None)
        async def run(body: WorkflowRunRequest):
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
        self, workflow_id: str, **configuration: object
    ) -> dict[str, object]:
        return {
            "inputs": {
                "slide": {"kind": "shared_path", "path": "slide.tiff"},
                "region_of_interest": {"kind": "shared_path", "path": "mask.tiff"},
            },
            "resolved_configuration": self.catalog.resolve_configuration(
                workflow_id, configuration
            ),
            "input_access": {
                "kind": "shared_storage",
                "state": {"root": str(self.root)},
            },
            "deadline_at": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
        }

    async def test_episeg_publishes_float_scores_without_heatmap_quantization(
        self,
    ) -> None:
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
        import h5py
        import lz4.frame

        pixels = np.full((224, 448, 3), 80, dtype=np.uint8)
        write_slide(self.root / "slide.tiff", pixels)
        write_slide(self.root / "mask.tiff", np.full_like(pixels, 255))

        async def model(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/prov-gigapath")
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
        import h5py
        import lz4.frame

        async def model(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/virchow2")
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
                        "body": self.invocation(workflow_id),
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
        import h5py

        write_slide(self.root / "mask.tiff", np.zeros_like(self.pixels))

        def reject_model_call(_request):
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
        entered = asyncio.Event()
        stopped = asyncio.Event()

        async def model(_request):
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
                "body": self.invocation("gigapath-embeddings"),
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
        import lz4.frame

        pixels = np.full((224, 448, 3), 80, dtype=np.uint8)
        write_slide(self.root / "slide.tiff", pixels)
        write_slide(self.root / "mask.tiff", np.full_like(pixels, 255))
        (self.root / "embeddings.h5").write_bytes(b"previous complete output")
        sizes = iter((3, 4))

        def model(_request):
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
        original = (self.root / "slide.tiff").read_bytes()
        client = await self.client("gigapath-embeddings")
        response = await client.post(
            "/v1/workflows/gigapath-embeddings/run",
            json=self.invocation("gigapath-embeddings", output_path="slide.tiff"),
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual((self.root / "slide.tiff").read_bytes(), original)
