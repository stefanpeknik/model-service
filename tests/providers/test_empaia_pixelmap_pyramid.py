"""Scores kept by the builder are published as a pixelmap pyramid, exactly."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from rationai.empaia.typing import SlideInfo

from model_service.builders.contracts import SharedArtifact
from model_service.builders.score_grid import (
    MEDIA_TYPE,
    ScoreGrid,
    write_score_grid,
)
from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.http_gateway import HttpEmpaiaGateway
from model_service.providers.empaia.image_io import ScoreGridPublisher
from tests.providers.empaia_fixtures import SLIDE_ID


def slide_info() -> SlideInfo:
    """Six levels from 384x256, each half the size of the one before."""
    levels = [
        {
            "extent": {"x": 384 // 2**level, "y": 256 // 2**level, "z": 1},
            "downsample_factor": float(2**level),
        }
        for level in range(6)
    ]
    return SlideInfo.model_validate(
        {
            "id": SLIDE_ID,
            "extent": {"x": 384, "y": 256, "z": 1},
            "num_levels": 6,
            "pixel_size_nm": {"x": 250.0, "y": 250.0},
            "tile_extent": {"x": 256, "y": 256, "z": 1},
            "levels": levels,
            "format": "test",
            "tissue": None,
            "stain": None,
        }
    )


class PyramidGateway:
    job_id = "job-1"

    def __init__(self) -> None:
        self.published: list[tuple[str, str, tuple[object, ...]]] = []

    async def slide_info(self) -> SlideInfo:
        return slide_info()

    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[object, ...]
    ) -> None:
        self.published.append((name, slide_id, tile_sets))


def grid(values: np.ndarray, scored: np.ndarray) -> ScoreGrid:
    # 64 px per score at level 0: 4 px at level 4, 2 px at level 5.
    return ScoreGrid(
        values=values,
        scored=scored,
        level=0,
        level_size=(384, 256),
        cell_size=64.0,
        mpp=(0.25, 0.25),
    )


def artifact(directory: str, scores: ScoreGrid) -> SharedArtifact:
    path = Path(directory) / "heatmap.npz"
    write_score_grid(path, scores)
    return SharedArtifact(path=str(path), media_type=MEDIA_TYPE)


class PixelmapPyramidTests(unittest.IsolatedAsyncioTestCase):
    async def publish(self, shared: SharedArtifact) -> PyramidGateway:
        gateway = PyramidGateway()
        await ScoreGridPublisher(tile_size=256).publish(
            gateway,  # type: ignore[arg-type]
            "heatmap",
            shared,
        )
        return gateway

    async def test_scores_are_published_exactly_at_every_level_from_the_coarsest_that_keeps_four_pixels_per_score(
        self,
    ) -> None:
        values = np.zeros((1, 4, 6), dtype=np.float32)
        scored = np.zeros((4, 6), dtype=bool)
        values[0, 1, 2], scored[1, 2] = 0.75, True
        values[0, 3, 5], scored[3, 5] = 0.25, True
        with TemporaryDirectory() as directory:
            gateway = await self.publish(artifact(directory, grid(values, scored)))

        [(name, slide_id, tile_sets)] = gateway.published
        self.assertEqual((name, slide_id), ("heatmap", SLIDE_ID))
        self.assertEqual([tile_set.level for tile_set in tile_sets], [4, 5])  # type: ignore[attr-defined]
        finest = tile_sets[0]
        [tile] = finest.tiles  # type: ignore[attr-defined]
        data = np.asarray(tile.data)
        self.assertEqual(data[4:8, 8:12].tolist(), [[0.75] * 4] * 4)
        self.assertEqual(data[12:16, 20:24].tolist(), [[0.25] * 4] * 4)
        # Cells the builder never scored are 0 in the tile, not scores.
        self.assertEqual(float(data[0:4, 0:4].max()), 0.0)
        self.assertEqual(float(data.max()), 0.75)

    async def test_grids_it_cannot_publish_are_invalid(self) -> None:
        nothing_scored = grid(np.zeros((1, 4, 6), np.float32), np.zeros((4, 6), bool))
        two_channels = grid(np.zeros((2, 4, 6), np.float32), np.ones((4, 6), bool))
        with TemporaryDirectory() as directory:
            unreadable = Path(directory) / "unreadable.npz"
            unreadable.write_bytes(b"not an npz")
            cases = {
                "nothing scored": artifact(directory, nothing_scored),
                "two channels": artifact(directory, two_channels),
                "unreadable": SharedArtifact(
                    path=str(unreadable), media_type=MEDIA_TYPE
                ),
                "not a grid": SharedArtifact(
                    path=str(unreadable), media_type="image/tiff"
                ),
            }
            for case, shared in cases.items():
                with (
                    self.subTest(case),
                    self.assertRaises(EmpaiaGatewayError) as raised,
                ):
                    await self.publish(shared)
                self.assertFalse(raised.exception.retryable)
                self.assertFalse(raised.exception.outcome_unknown)


class PyramidUploadFailureTests(unittest.IsolatedAsyncioTestCase):
    def gateway(self, upload: object) -> HttpEmpaiaGateway:
        client = SimpleNamespace(upload_pixelmap_pyramid=upload)
        return HttpEmpaiaGateway(
            "https://empaia.invalid", "job-1", "credential", client=client
        )  # type: ignore[arg-type]

    async def test_a_pyramid_the_sdk_gave_up_on_is_an_unknown_outcome(self) -> None:
        async def gave_up(**_kwargs: object) -> str:
            raise RuntimeError("Pixelmap tile upload failed after 5 attempts")

        with self.assertRaises(EmpaiaGatewayError) as raised:
            await self.gateway(gave_up).publish_pixelmap("heatmap", SLIDE_ID, ())

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertFalse(raised.exception.retryable)

    async def test_a_direct_write_that_failed_is_an_unknown_outcome(self) -> None:
        async def failed(**_kwargs: object) -> str:
            raise PermissionError("pixelmap storage is not writable")

        with self.assertRaises(EmpaiaGatewayError) as raised:
            await self.gateway(failed).publish_pixelmap("heatmap", SLIDE_ID, ())

        self.assertTrue(raised.exception.outcome_unknown)
        self.assertFalse(raised.exception.retryable)

    async def test_the_pixelmap_is_created_under_the_output_key_for_the_slide(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        async def uploaded(**kwargs: object) -> str:
            calls.append(kwargs)
            return "pixelmap-id"

        await self.gateway(uploaded).publish_pixelmap("heatmap", SLIDE_ID, ())

        self.assertEqual(
            calls,
            [
                {
                    "key": "heatmap",
                    "name": "heatmap",
                    "wsi_id": SLIDE_ID,
                    "tile_sets": (),
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
