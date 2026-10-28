"""Check tile placement, overlap means, padding and masks with native image libraries."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from model_service.builders.contracts import SpatialScoresModel
from model_service.builders.heatmap.processor import (
    HeatmapRequest,
    SlideHeatmapProcessor,
)
from model_service.builders.registry import ScoreEncoding
from model_service.builders.score_grid import (
    ScoreGrid,
    read_score_grid,
    write_score_grid,
)
from tests.support.builder_fixtures import write_slide


class LandmarkPredictor:
    """Return a constant score, or the tile's origin pixel, to expose placement errors."""

    def __init__(
        self, output_tile_size: int = 1, constant: float | None = None
    ) -> None:
        self.output_tile_size = output_tile_size
        self.constant = constant
        self.calls = 0

    async def predict(self, tile: object) -> np.ndarray:
        self.calls += 1
        score = self.constant
        if score is None:
            score = float(np.asarray(tile)[0, 0, 0]) / 255
        return np.full((self.output_tile_size, self.output_tile_size), score)


class HeatmapGeometryTests(unittest.IsolatedAsyncioTestCase):
    async def _build(
        self,
        root: Path,
        width: int,
        height: int,
        *,
        stride_fraction: float = 0.5,
        predictor: LandmarkPredictor | None = None,
        mask: str = "full",
        encoding: ScoreEncoding = ScoreEncoding.UINT8,
        stale_grid: bool = False,
    ) -> np.ndarray:
        import pyvips

        predictor = predictor or LandmarkPredictor()
        directory = root / f"{width}-{height}-{mask}"
        directory.mkdir()
        if stale_grid:
            write_score_grid(
                directory / "heatmap.tiff",
                ScoreGrid(
                    np.zeros((2, 2), np.float32), np.ones((2, 2), bool), 16.0, (4, 4)
                ),
            )
        y, x = np.indices((height, width))
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = ((x // 16 + y // 16) % 2) * 255
        write_slide(directory / "slide.tiff", pixels)
        mask_path = None
        if mask != "none":
            mask_path = directory / "mask.tiff"
            write_slide(mask_path, np.full_like(pixels, 0 if mask == "empty" else 255))
        output = directory / "heatmap.tiff"
        await SlideHeatmapProcessor(
            predictor, num_threads=1, max_concurrent_tasks=2
        ).build(
            SpatialScoresModel(
                id="landmark",
                tile_size=32,
                output_tile_size=predictor.output_tile_size,
                n_channels=1,
                mpp=0.5,
            ),
            HeatmapRequest(
                slide_path=directory / "slide.tiff",
                mask_path=mask_path,
                output_path=output,
                stride_fraction=stride_fraction,
                output_tile_height=128,
                output_tile_width=128,
                encoding=encoding,
            ),
        )
        image = pyvips.Image.new_from_file(str(output))
        self.assertEqual((image.width, image.height, image.bands), (width, height, 1))
        self.assertAlmostEqual(image.xres, 2000)
        self.assertAlmostEqual(image.yres, 2000)
        floating = encoding is ScoreEncoding.FLOAT32
        self.assertEqual(image.format, "float" if floating else "uchar")
        return np.frombuffer(
            image.write_to_memory(), dtype=np.float32 if floating else np.uint8
        ).reshape(height, width)

    async def test_padding_is_cropped_without_compressing_tile_placement(self) -> None:
        for width, height, stride, output_tile_size in (
            (112, 80, 0.5, 1),
            (128, 96, 1.0, 1),
            (112, 82, 0.3, 1),
            (112, 80, 0.5, 2),
            (112, 80, 0.5, 3),
            (112, 80, 0.5, 64),
        ):
            with (
                self.subTest(stride=stride, output_tile_size=output_tile_size),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                aligned = await self._build(
                    root,
                    width,
                    height,
                    stride_fraction=stride,
                    predictor=LandmarkPredictor(output_tile_size),
                )
                unaligned = await self._build(
                    root,
                    107,
                    75,
                    stride_fraction=stride,
                    predictor=LandmarkPredictor(output_tile_size),
                )
                np.testing.assert_array_equal(unaligned, aligned[:75, :107])

    async def test_the_scores_are_kept_beside_the_tiff(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            constant = LandmarkPredictor(constant=0.5)
            await self._build(root, 112, 80, predictor=constant)
            await self._build(root, 112, 80, predictor=constant, mask="empty")
            scored = read_score_grid(root / "112-80-full" / "heatmap.tiff")
            unscored = read_score_grid(root / "112-80-empty" / "heatmap.tiff")

        assert scored is not None and unscored is not None
        # A tile every 16 px yields one score per 16 px; the empty mask scores none.
        self.assertEqual((scored.cell_size, scored.level_size), (16.0, (112, 80)))
        self.assertTrue(scored.scored.any())
        np.testing.assert_allclose(scored.values[scored.scored], 0.5)
        self.assertFalse(unscored.scored.any())

    async def test_a_full_resolution_output_has_no_score_grid(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            await self._build(
                root,
                112,
                80,
                predictor=LandmarkPredictor(output_tile_size=32),
                stale_grid=True,
            )

            self.assertIsNone(read_score_grid(root / "112-80-full" / "heatmap.tiff"))

    async def test_aligned_output_preserves_original_pixels_and_overlap_means(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            pixels = await self._build(Path(directory), 112, 80)

        self.assertEqual(
            hashlib.sha256(pixels.tobytes()).hexdigest(),
            "01a90dd369c6c5b5122335061de7cc609156cb094798a5d3851cc98a2333b58f",
        )
        self.assertEqual(int(pixels[0, 0]), 0)
        self.assertEqual(int(pixels[0, -1]), 255)
        self.assertEqual(int(pixels[-1, 0]), 255)
        self.assertEqual(int(pixels[-1, -1]), 0)
        self.assertEqual(int(pixels[39, 55]), 127)

    async def test_constant_scores_remain_constant_through_overlap_and_padding(
        self,
    ) -> None:
        predictor = LandmarkPredictor(constant=0.75)
        with TemporaryDirectory() as directory:
            pixels = await self._build(Path(directory), 107, 75, predictor=predictor)

        self.assertEqual(predictor.calls, 24)
        np.testing.assert_array_equal(pixels, np.full((75, 107), 191, dtype=np.uint8))

    async def test_float32_encoding_keeps_unquantized_scores(self) -> None:
        with TemporaryDirectory() as directory:
            pixels = await self._build(
                Path(directory),
                107,
                75,
                predictor=LandmarkPredictor(constant=0.375),
                encoding=ScoreEncoding.FLOAT32,
            )
        np.testing.assert_allclose(pixels, 0.375)

    async def test_without_a_mask_every_grid_tile_is_predicted(self) -> None:
        masked = LandmarkPredictor()
        unmasked = LandmarkPredictor()
        with TemporaryDirectory() as directory:
            with_mask = await self._build(Path(directory), 107, 75, predictor=masked)
            without_mask = await self._build(
                Path(directory), 107, 75, predictor=unmasked, mask="none"
            )
        self.assertEqual(unmasked.calls, 24)
        self.assertEqual(masked.calls, 24)
        np.testing.assert_array_equal(without_mask, with_mask)

    async def test_empty_mask_skips_predictions_and_writes_zero_background(
        self,
    ) -> None:
        predictor = LandmarkPredictor()
        with TemporaryDirectory() as directory:
            pixels = await self._build(
                Path(directory), 107, 75, predictor=predictor, mask="empty"
            )

        self.assertEqual(predictor.calls, 0)
        self.assertFalse(pixels.any())


if __name__ == "__main__":
    unittest.main()
