"""Find tissue on a slide's coarse level so builders can skip empty glass.

Colour saturation separates stained tissue from white glass and from the black
that transparent regions (e.g. in MIRAX slides) read as. Otsu's method picks the
threshold per slide, with a floor so a slide of nearly only glass is not split
into "tissue" by noise. This is a stopgap until QC publishes tissue masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ratiopath.openslide import OpenSlide


TISSUE_MPP = 8.0
MIN_SATURATION = 0.05
MAX_PIXELS = 64_000_000


@dataclass(frozen=True, slots=True)
class TissueMask:
    """Tissue pixels of one slide level, which spans the whole slide."""

    pixels: NDArray[np.bool_]

    @property
    def fraction(self) -> float:
        return float(self.pixels.mean())


def find_tissue(slide: OpenSlide) -> TissueMask | None:
    """Tissue on the level closest to `TISSUE_MPP`; None when no level is small enough."""
    if (
        PROPERTY_NAME_MPP_X in slide.properties
        and PROPERTY_NAME_MPP_Y in slide.properties
    ):
        level = slide.closest_level(TISSUE_MPP)
    else:
        level = slide.level_count - 1
    width, height = slide.level_dimensions[level]
    if width * height > MAX_PIXELS:
        return None
    rgb = np.asarray(slide.read_region((0, 0), level, (width, height)))[..., :3]
    return TissueMask(tissue_pixels(rgb))


def tissue_pixels(rgb: NDArray[np.uint8]) -> NDArray[np.bool_]:
    brightest = rgb.max(axis=-1).astype(np.float32)
    darkest = rgb.min(axis=-1).astype(np.float32)
    saturation = _box_blur(
        np.divide(
            brightest - darkest,
            brightest,
            out=np.zeros_like(brightest),
            where=brightest > 0,
        )
    )
    return saturation > max(_otsu(saturation), MIN_SATURATION)


def _box_blur(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """3x3 mean, so single noisy glass pixels do not pass as tissue."""
    padded = np.pad(values, 1, mode="edge")
    height, width = values.shape
    total = np.zeros_like(values)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + height, dx : dx + width]
    total /= 9
    return total


def _otsu(values: NDArray[np.float32]) -> float:
    """The threshold in [0, 1] that best separates two classes of `values`."""
    histogram, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    weight_low = np.cumsum(histogram)
    weight_high = weight_low[-1] - weight_low
    sum_low = np.cumsum(histogram * centers)
    mean_low = np.divide(
        sum_low, weight_low, out=np.zeros_like(sum_low), where=weight_low > 0
    )
    mean_high = np.divide(
        sum_low[-1] - sum_low,
        weight_high,
        out=np.zeros_like(sum_low),
        where=weight_high > 0,
    )
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    return float(centers[int(np.argmax(between))])
