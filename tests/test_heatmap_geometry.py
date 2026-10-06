from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from workflows.heatmap.processor import (
    HeatmapBuildRequest,
    ModelTileConfiguration,
    SlideHeatmapProcessor,
)


class LandmarkPredictor:
    def __init__(
        self, output_tile_size: int = 1, constant: float | None = None
    ) -> None:
        self.output_tile_size = output_tile_size
        self.constant = constant
        self.calls = 0

    async def configuration(self) -> ModelTileConfiguration:
        return ModelTileConfiguration(32, self.output_tile_size, 1, 0.5)

    async def predict(self, tile: object) -> np.ndarray:
        self.calls += 1
        score = self.constant
        if score is None:
            score = float(np.asarray(tile)[0, 0, 0]) / 255
        return np.full((self.output_tile_size, self.output_tile_size), score)


@unittest.skipUnless(
    os.environ.get("RUN_NATIVE_HEATMAP_TESTS") == "1",
    "native heatmap regression profile is not enabled",
)
class HeatmapGeometryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _write_slide(path: Path, pixels: np.ndarray) -> None:
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

    async def _build(
        self,
        root: Path,
        width: int,
        height: int,
        *,
        stride_fraction: float = 0.5,
        predictor: LandmarkPredictor | None = None,
        empty_mask: bool = False,
    ) -> np.ndarray:
        import pyvips

        directory = root / f"{width}-{height}"
        directory.mkdir()
        y, x = np.indices((height, width))
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = ((x // 16 + y // 16) % 2) * 255
        self._write_slide(directory / "slide.tiff", pixels)
        self._write_slide(
            directory / "mask.tiff", np.full_like(pixels, 0 if empty_mask else 255)
        )
        output = directory / "heatmap.tiff"
        await SlideHeatmapProcessor(
            predictor or LandmarkPredictor(),
            num_threads=1,
            max_concurrent_tasks=2,
        ).build(
            HeatmapBuildRequest(
                slide_path=directory / "slide.tiff",
                tissue_mask_path=directory / "mask.tiff",
                output_path=output,
                stride_fraction=stride_fraction,
                output_tile_height=128,
                output_tile_width=128,
            )
        )
        image = pyvips.Image.new_from_file(str(output))
        self.assertEqual((image.width, image.height, image.bands), (width, height, 1))
        self.assertAlmostEqual(image.xres, 2000)
        self.assertAlmostEqual(image.yres, 2000)
        return np.frombuffer(image.write_to_memory(), dtype=np.uint8).reshape(
            height, width
        )

    async def test_padding_is_cropped_without_compressing_tile_placement(self) -> None:
        for width, height, stride, output_tile_size in (
            (112, 80, 0.5, 1),
            (128, 96, 1.0, 1),
            (112, 82, 0.3, 1),
            (112, 80, 0.5, 2),
            (112, 80, 0.5, 3),
            (112, 80, 0.5, 64),
        ):
            with self.subTest(stride=stride, output_tile_size=output_tile_size):
                with TemporaryDirectory() as directory:
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

    async def test_empty_mask_skips_predictions_and_writes_zero_background(
        self,
    ) -> None:
        predictor = LandmarkPredictor()
        with TemporaryDirectory() as directory:
            pixels = await self._build(
                Path(directory),
                107,
                75,
                predictor=predictor,
                empty_mask=True,
            )

        self.assertEqual(predictor.calls, 0)
        self.assertFalse(pixels.any())
