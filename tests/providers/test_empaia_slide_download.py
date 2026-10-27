"""Slides downloaded through the App API read like the original level they came from."""

from __future__ import annotations

import io
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvips
from PIL import Image
from ratiopath.openslide import OpenSlide

from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.http_gateway import (
    Rectangle,
    SlideLevel,
    SlideMetadata,
)
from model_service.providers.empaia.image_io import TiffSlideDownloader, closest_level


WIDTH, HEIGHT = 1200, 700


def pyramid() -> list[np.ndarray]:
    """Three levels of a smooth RGB image: level 0, halved, and an eighth."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    base = np.stack([x % 256, y % 256, (x + y) // 8 % 256], axis=-1).astype(np.uint8)
    return [base, base[::2, ::2], base[::8, ::8]]


SLIDE = SlideMetadata(
    identifier="slide-1",
    width=WIDTH,
    height=HEIGHT,
    pixel_size_nm_x=250.0,
    pixel_size_nm_y=250.0,
    levels=(
        SlideLevel(index=0, width=WIDTH, height=HEIGHT),
        SlideLevel(index=1, width=WIDTH // 2, height=HEIGHT // 2),
    ),
)


class RegionServer:
    """Serves PNG regions of `pyramid()` like the App API's region route."""

    def __init__(self, fail_at: tuple[int, int] | None = None) -> None:
        self.levels = pyramid()
        self.fail_at = fail_at
        self.requests: list[tuple[str, int, int, int, int, int]] = []

    async def read_region(
        self, slide_id: str, level: int, x: int, y: int, width: int, height: int
    ) -> bytes:
        self.requests.append((slide_id, level, x, y, width, height))
        if (x, y) == self.fail_at:
            raise EmpaiaGatewayError("unavailable", retryable=True)
        buffer = io.BytesIO()
        region = self.levels[level][y : y + height, x : x + width]
        Image.fromarray(region).save(buffer, format="PNG")
        return buffer.getvalue()


class SlideDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_level_closest_to_the_model_is_written_with_its_resolution(
        self,
    ) -> None:
        server = RegionServer()
        with TemporaryDirectory() as directory:
            target = Path(directory) / "slide.tiff"

            await TiffSlideDownloader(tile_size=256, concurrency=3).download(
                server,  # type: ignore[arg-type]
                SLIDE,
                0.48,
                target,
            )

            with OpenSlide(str(target)) as slide:
                self.assertEqual(slide.level_dimensions, ((600, 350),))
                self.assertAlmostEqual(float(slide.properties["openslide.mpp-x"]), 0.5)
                self.assertEqual(slide.closest_level(0.48), 0)
                pixels = np.asarray(slide.read_region((0, 0), 0, (600, 350)))[..., :3]
        self.assertEqual({request[1] for request in server.requests}, {1})
        # 600x350 in 256-px tiles: 3 columns, 2 rows, edges clipped to the level.
        self.assertEqual(len(server.requests), 6)
        self.assertIn(("slide-1", 1, 512, 256, 88, 94), server.requests)
        difference = np.abs(pixels.astype(int) - pyramid()[1].astype(int))
        self.assertLess(difference.mean(), 3)

    async def test_only_tiles_near_the_region_are_fetched(self) -> None:
        server = RegionServer()
        with TemporaryDirectory() as directory:
            target = Path(directory) / "slide.tiff"

            # Level 1 box (50, 50)-(150, 100), widened by 16 px: inside tile (0, 0).
            await TiffSlideDownloader(tile_size=256).download(
                server,  # type: ignore[arg-type]
                SLIDE,
                0.48,
                target,
                region=Rectangle(100, 100, 200, 100),
                margin=16,
            )

            with OpenSlide(str(target)) as slide:
                pixels = np.asarray(slide.read_region((0, 0), 0, (600, 350)))[..., :3]
        self.assertEqual(server.requests, [("slide-1", 1, 0, 0, 256, 256)])
        fetched = pixels[:256, :256].astype(int) - pyramid()[1][:256, :256]
        self.assertLess(np.abs(fetched).mean(), 3)
        # Skipped tiles are white, which the builder treats as background.
        self.assertGreater(pixels[:, 256:].min(), 250)
        self.assertGreater(pixels[256:, :].min(), 250)

    async def test_a_coarse_level_is_appended_for_finding_tissue(self) -> None:
        server = RegionServer()
        slide = replace(
            SLIDE,
            levels=(
                *SLIDE.levels,
                SlideLevel(index=2, width=WIDTH // 8, height=-(-HEIGHT // 8)),
            ),
        )
        with TemporaryDirectory() as directory:
            target = Path(directory) / "slide.tiff"

            await TiffSlideDownloader(tile_size=256).download(
                server,  # type: ignore[arg-type]
                slide,
                0.48,
                target,
            )

            with OpenSlide(str(target)) as downloaded:
                self.assertEqual(downloaded.level_dimensions, ((600, 350), (150, 88)))
                self.assertEqual(downloaded.closest_level(0.48), 0)
                self.assertEqual(downloaded.closest_level(8.0), 1)
                overview = np.asarray(downloaded.read_region((0, 0), 1, (150, 88)))
        difference = np.abs(overview[..., :3].astype(int) - pyramid()[2].astype(int))
        self.assertLess(difference.mean(), 4)
        self.assertEqual({request[1] for request in server.requests}, {1, 2})

    async def test_failed_region_leaves_no_partial_slide(self) -> None:
        server = RegionServer(fail_at=(256, 256))
        with TemporaryDirectory() as directory:
            target = Path(directory) / "slide.tiff"

            with self.assertRaises(EmpaiaGatewayError):
                await TiffSlideDownloader(tile_size=256, concurrency=2).download(
                    server,  # type: ignore[arg-type]
                    SLIDE,
                    0.48,
                    target,
                )

            self.assertFalse(target.exists())

    def test_level_choice_matches_openslide_on_the_original_slide(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.tiff"
            pyvips.Image.new_from_array(pyramid()[0]).tiffsave(
                str(source),
                tile=True,
                pyramid=True,
                tile_width=256,
                tile_height=256,
                xres=1_000_000 / SLIDE.pixel_size_nm_x,
                yres=1_000_000 / SLIDE.pixel_size_nm_y,
            )
            with OpenSlide(str(source)) as original:
                levels = tuple(
                    SlideLevel(index=index, width=width, height=height)
                    for index, (width, height) in enumerate(original.level_dimensions)
                )
                slide = SlideMetadata("s", WIDTH, HEIGHT, 250.0, 250.0, levels)
                for mpp in (0.2, 0.25, 0.37, 0.38, 0.5, 0.9, 4.0):
                    with self.subTest(mpp=mpp):
                        self.assertEqual(
                            closest_level(slide, mpp).index,
                            original.closest_level(mpp),
                        )


if __name__ == "__main__":
    unittest.main()
