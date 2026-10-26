"""Pixelmap encoding, tile-upload retries, URL allowlist and workspace cleanup."""

from __future__ import annotations

import json
import struct
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from model_service.builders.contracts import (
    SCORE_SPACING_KEY,
    BuilderResult,
    SharedArtifact,
)
from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.http_gateway import HttpEmpaiaGateway
from model_service.providers.empaia.image_io import TiffPixelmapPublisher
from model_service.providers.empaia.input_access import job_workspace
from model_service.providers.empaia.provider import EmpaiaProviderAdapter
from model_service.scheduler.contracts import (
    ExecutionError,
    InvalidScheduleRequestError,
    ScheduleRequest,
)
from model_service.scheduler.providers import (
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderSuccess,
)


try:
    import pyvips
except (ImportError, OSError):
    HAS_VIPS = False
else:
    HAS_VIPS = bool(pyvips.Image)


class RecordingGateway:
    job_id = "job-1"

    def __init__(self, width: int, height: int, levels: int = 1) -> None:
        self.created: list[dict[str, object]] = []
        self.tiles: list[bytes] = []
        self.positions: list[tuple[int, int, int]] = []
        self._slide = SimpleNamespace(
            identifier="slide-1",
            levels=tuple(
                SimpleNamespace(
                    index=index, width=width >> index, height=height >> index
                )
                for index in range(levels)
            ),
        )

    async def slide_metadata(self) -> object:
        return self._slide

    async def create_pixelmap(self, _name: str, body: dict[str, object]) -> str:
        self.created.append(body)
        return "pixelmap-1"

    async def put_pixelmap_tile(
        self, _pixelmap: str, level: int, tile_x: int, tile_y: int, data: bytes
    ) -> None:
        self.positions.append((level, tile_x, tile_y))
        self.tiles.append(data)


def floats(data: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(data) // 4}f", data))


@unittest.skipUnless(HAS_VIPS, "libvips is unavailable")
class PixelmapEncodingTests(unittest.IsolatedAsyncioTestCase):
    async def publish(self, image: pyvips.Image) -> RecordingGateway:
        gateway = RecordingGateway(image.width, image.height)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scores.tiff"
            image.tiffsave(str(path))
            await TiffPixelmapPublisher(tile_size=256).publish(
                gateway,  # type: ignore[arg-type]
                "heatmap",
                SharedArtifact(path=str(path), media_type="image/tiff"),
            )
        return gateway

    async def test_heatmap_goes_to_the_coarsest_level_that_keeps_every_score(
        self,
    ) -> None:
        # A 4x6 grid of distinct scores, upsampled 64x like the heatmap builder does.
        scores = np.arange(24, dtype=np.float32).reshape(4, 6) / 24 + 0.01
        image = pyvips.Image.new_from_array(scores).resize(64)
        image = (image * 255).cast("uchar")
        image.set_type(
            pyvips.GValue.gstr_type,
            "image-description",
            json.dumps({SCORE_SPACING_KEY: 64.0}),
        )
        # Slide levels 384x256 down to 24x16 (16x): 4 pixels per score there.
        gateway = RecordingGateway(image.width, image.height, levels=6)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scores.tiff"
            image.tiffsave(str(path), tile=True, pyramid=True)
            await TiffPixelmapPublisher(tile_size=256).publish(
                gateway,  # type: ignore[arg-type]
                "heatmap",
                SharedArtifact(path=str(path), media_type="image/tiff"),
            )

        level = gateway.created[0]["levels"][0]  # type: ignore[index]
        self.assertEqual(level["slide_level"], 4)
        self.assertEqual(gateway.positions, [(4, 0, 0)])
        published = np.asarray(floats(gateway.tiles[0])).reshape(256, 256)[:16, :24]
        # Each published pixel is the full-resolution heatmap averaged over its area.
        full = (
            np.ndarray(buffer=image.write_to_memory(), dtype=np.uint8, shape=(256, 384))
            / 255
        )
        np.testing.assert_allclose(
            published, full.reshape(16, 16, 24, 16).mean(axis=(1, 3)), atol=0.01
        )

    async def test_uint8_scores_are_scaled_to_unit_range(self) -> None:
        gateway = await self.publish(pyvips.Image.black(4, 4).new_from_image(51))

        self.assertEqual(gateway.created[0]["element_type"], "float32")
        values = floats(gateway.tiles[0])
        self.assertAlmostEqual(values[0], 0.2, places=6)
        # The edge tile is padded with zeros up to the provider tile size.
        self.assertEqual(len(values), 256 * 256)
        self.assertEqual(values[4], 0.0)

    async def test_float32_scores_are_published_unchanged(self) -> None:
        image = (pyvips.Image.black(4, 4) + 0.375).cast("float")

        values = floats((await self.publish(image)).tiles[0])

        self.assertEqual(values[0], 0.375)

    async def test_channels_are_sent_as_separate_planes(self) -> None:
        image = (
            pyvips.Image.black(4, 4).bandjoin_const([0]).linear([1, 1], [0.25, 0.75])
        )
        image = image.cast("float")

        gateway = await self.publish(image)

        values = floats(gateway.tiles[0])
        self.assertEqual(gateway.created[0]["channel_count"], 2)
        self.assertEqual(values[0], 0.25)
        self.assertEqual(values[256 * 256], 0.75)

    async def test_other_pixel_formats_are_rejected(self) -> None:
        with self.assertRaises(EmpaiaGatewayError):
            await self.publish(pyvips.Image.black(4, 4).cast("ushort"))


class TileUploadRetryTests(unittest.IsolatedAsyncioTestCase):
    def gateway(self, put_tile: MagicMock) -> HttpEmpaiaGateway:
        client = SimpleNamespace(put_tile=put_tile)
        return HttpEmpaiaGateway(
            "https://empaia.invalid", "job-1", "credential", client=client
        )  # type: ignore[arg-type]

    async def test_transient_tile_failure_is_retried(self) -> None:
        unavailable = EmpaiaGatewayError("unavailable", retryable=True)
        gateway = self.gateway(MagicMock(return_value=None))
        calls = 0

        async def flaky(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise unavailable

        with (
            patch.object(gateway, "_call", side_effect=flaky),
            patch("asyncio.sleep", AsyncMock()),
        ):
            await gateway.put_pixelmap_tile("pixelmap-1", 0, 0, 0, b"tile")

        self.assertEqual(calls, 2)

    async def test_permanent_tile_failure_is_not_retried(self) -> None:
        gateway = self.gateway(MagicMock(return_value=None))
        rejected = EmpaiaGatewayError("rejected", retryable=False)

        with (
            patch.object(gateway, "_call", side_effect=rejected) as call,
            self.assertRaises(EmpaiaGatewayError),
        ):
            await gateway.put_pixelmap_tile("pixelmap-1", 0, 0, 0, b"tile")

        self.assertEqual(call.call_count, 1)


def unused_gateway_factory(_context: object) -> object:
    raise AssertionError("the gateway must not be created")


class ProviderSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_unlisted_appservice_never_receives_the_credential(self) -> None:
        provider = EmpaiaProviderAdapter(
            unused_gateway_factory,  # type: ignore[arg-type]
            allowed_api_urls=("https://empaia.example/app-api",),
        )
        for api_url in (
            "https://attacker.example",
            "https://empaia.example/app-api-evil",
            "https://empaia.example",
        ):
            with (
                self.subTest(api_url=api_url),
                self.assertRaises(InvalidScheduleRequestError),
            ):
                await provider.prepare(
                    ScheduleRequest(
                        builder_id="heatmap",
                        model_id="prostate-classifier-1",
                        provider="empaia",
                        body={
                            "installation_id": "installation-1",
                            "api_url": api_url,
                            "job_id": "job-1",
                            "credential": "token",
                        },
                    ),
                    datetime.now(UTC) + timedelta(hours=1),
                )

    def test_an_allowlist_is_required(self) -> None:
        with self.assertRaises(ValueError):
            EmpaiaProviderAdapter(unused_gateway_factory, allowed_api_urls=())  # type: ignore[arg-type]

    def job_workspaces(self, root: Path) -> tuple[Path, Path, Path]:
        """Two attempts of job "j" (one with a heatmap) and another job's workspace."""
        job = job_workspace(root, "https://empaia.invalid", "j")
        artifact = job / "attempt-2" / "outputs" / "heatmap.tiff"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"tiff")
        (job / "attempt-1" / "outputs").mkdir(parents=True)
        other_job = job_workspace(root, "https://empaia.invalid", "other")
        other_job.mkdir()
        return job, artifact, other_job

    async def test_every_attempt_workspace_is_removed_after_publication(
        self,
    ) -> None:
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job, artifact, other_job = self.job_workspaces(root)
            provider = EmpaiaProviderAdapter(
                lambda _context: gateway,  # type: ignore[arg-type, return-value]
                allowed_api_urls=("https://empaia.invalid",),
                workspace_root=root,
            )

            await provider.finish(
                {"api_url": "https://empaia.invalid", "job_id": "j", "credential": "c"},
                ProviderSuccess(
                    BuilderResult(
                        outputs={
                            "heatmap": SharedArtifact(
                                path=str(artifact), media_type="image/tiff"
                            )
                        }
                    )
                ),
            )

            gateway.finalize.assert_awaited_once()
            self.assertFalse(job.exists())
            self.assertTrue(other_job.exists())

    async def test_failed_jobs_free_their_workspace_even_if_reporting_fails(
        self,
    ) -> None:
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        gateway.report_failure.side_effect = EmpaiaGatewayError(
            "EMPAIA AppService is temporarily unavailable", retryable=True
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job, _artifact, other_job = self.job_workspaces(root)
            provider = EmpaiaProviderAdapter(
                lambda _context: gateway,  # type: ignore[arg-type, return-value]
                allowed_api_urls=("https://empaia.invalid",),
                workspace_root=root,
            )

            with self.assertRaises(ProviderFinishFailureError):
                await provider.finish(
                    {
                        "api_url": "https://empaia.invalid",
                        "job_id": "j",
                        "credential": "c",
                    },
                    ProviderFailure(ExecutionError("builder_failed", "it failed")),
                )

            self.assertFalse(job.exists())
            self.assertTrue(other_job.exists())

    async def test_a_publication_that_fails_keeps_outputs_for_its_retry(
        self,
    ) -> None:
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        gateway.finalize.side_effect = EmpaiaGatewayError(
            "EMPAIA AppService is temporarily unavailable", retryable=True
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _job, artifact, _other = self.job_workspaces(root)
            provider = EmpaiaProviderAdapter(
                lambda _context: gateway,  # type: ignore[arg-type, return-value]
                allowed_api_urls=("https://empaia.invalid",),
                workspace_root=root,
            )

            with self.assertRaises(ProviderFinishFailureError):
                await provider.finish(
                    {
                        "api_url": "https://empaia.invalid",
                        "job_id": "j",
                        "credential": "c",
                    },
                    ProviderSuccess(BuilderResult(outputs={})),
                )

            self.assertTrue(artifact.exists())
