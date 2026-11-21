"""Tissue detection runs on the coarse level and filters tiles by tissue share."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from ratiopath.openslide import OpenSlide

from model_service.builders.slides import tissue
from model_service.builders.slides.tile_reader import TileReader
from model_service.contracts.failures import InvalidInvocation
from model_service.testing import fake_slide


def slide_pixels() -> np.ndarray:
    rng = np.random.default_rng(0)
    # Mild noise, as on a coarse pyramid level; multi-Otsu needs a few grey values.
    pixels = np.clip(242 + rng.normal(0, 0.5, (200, 300, 3)), 0, 255).astype(np.uint8)
    pixels[50:150, 50:150] = (190, 110, 170)  # H&E tissue
    pixels[60:70, 60:70] = (120, 60, 140)  # darker nucleus-rich patch
    return pixels


class TissueTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "slide.tiff"
        # 2400x1600 at 0.5 µm/px; the level closest to 8 µm/px is 16x smaller.
        big = np.kron(slide_pixels(), np.ones((8, 8, 1), dtype=np.uint8))
        fake_slide(self.path, big, mpp=0.5, pyramid=True)

    def test_mask_comes_from_the_coarse_level_and_finds_the_tissue(self) -> None:
        with OpenSlide(str(self.path)) as slide:
            mask = tissue.find_tissue(slide)

        self.assertEqual(mask.pixels.shape, (100, 150))
        self.assertTrue(mask.pixels[30:70, 30:70].all())
        self.assertFalse(mask.pixels[:20].any())

    def test_a_slide_without_a_coarse_level_is_refused(self) -> None:
        with (
            OpenSlide(str(self.path)) as slide,
            mock.patch.object(tissue, "MAX_PIXELS", 100),
            self.assertRaises(InvalidInvocation),
        ):
            tissue.find_tissue(slide)

    def test_min_tissue_fraction_skips_tiles_mostly_glass(self) -> None:
        # Level 0 tile at (300, 300) of 200 px: tissue (400..1200) covers a quarter of it.
        with (
            OpenSlide(str(self.path)) as slide,
            TileReader.open(slide, None, 0.5, skip_background=True) as any_tissue,
            TileReader.open(
                slide, None, 0.5, skip_background=True, min_tissue_fraction=0.6
            ) as mostly_tissue,
        ):
            self.assertIsNotNone(any_tissue.read(300, 300, 200))
            self.assertIsNone(mostly_tissue.read(300, 300, 200))
            self.assertIsNotNone(mostly_tissue.read(500, 500, 200))
            self.assertIsNone(any_tissue.read(0, 0, 200))


if __name__ == "__main__":
    unittest.main()
