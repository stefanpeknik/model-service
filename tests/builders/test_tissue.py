"""Tissue is told apart from white glass and transparent (black) regions."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ratiopath.openslide import OpenSlide

from model_service.builders.tissue import find_tissue, tissue_pixels
from tests.support.builder_fixtures import write_slide


def slide_pixels() -> np.ndarray:
    pixels = np.full((200, 300, 3), 242, dtype=np.uint8)  # glass
    pixels[:, 250:] = 0  # transparent MIRAX region
    pixels[50:150, 50:150] = (190, 110, 170)  # H&E tissue
    pixels[60:70, 60:70] = (120, 60, 140)  # darker nucleus-rich patch
    return pixels


class TissueTests(unittest.TestCase):
    def test_tissue_is_found_and_glass_and_black_are_not(self) -> None:
        tissue = tissue_pixels(slide_pixels())

        self.assertTrue(tissue[50:150, 50:150].all())
        self.assertFalse(tissue[:40].any())
        self.assertFalse(tissue[:, 250:].any())

    def test_glass_only_finds_no_tissue(self) -> None:
        rng = np.random.default_rng(0)
        glass = np.clip(242 + rng.normal(0, 2, (200, 300, 3)), 0, 255).astype(np.uint8)

        self.assertFalse(tissue_pixels(glass).any())

    def test_mask_comes_from_the_coarse_level(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "slide.tiff"
            big = np.kron(slide_pixels(), np.ones((8, 8, 1), dtype=np.uint8))
            write_slide(path, big, mpp=0.5, pyramid=True)
            with OpenSlide(str(path)) as slide:
                mask = find_tissue(slide)

        assert mask is not None
        # 2400x1600 at 0.5 µm/px; the closest level to 8 µm/px is 16x smaller.
        self.assertEqual(mask.pixels.shape, (100, 150))
        self.assertAlmostEqual(mask.fraction, 100 * 100 / (200 * 300), delta=0.02)


if __name__ == "__main__":
    unittest.main()
