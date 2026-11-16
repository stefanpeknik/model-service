"""Find tissue on a slide's coarse level so builders can skip empty glass.

Detection is ratiopath's `tissue_mask` (grayscale, multi-Otsu, closing and
opening). Transparent regions (e.g. in MIRAX slides) are flattened onto white
first, so they read as glass. This is a stopgap until QC publishes tissue masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyvips
from ratiopath.masks.tissue_mask import tissue_mask

from model_service.contracts.failures import InvalidInvocation


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ratiopath.openslide import OpenSlide


TISSUE_MPP = 8.0
# The level is read into memory; a slide without a coarse level would not fit.
MAX_PIXELS = 64_000_000


@dataclass(frozen=True, slots=True)
class TissueMask:
    """Tissue pixels of one slide level, which spans the whole slide.

    Attributes:
        pixels: True where a pixel is tissue.
    """

    pixels: NDArray[np.bool_]

    @property
    def fraction(self) -> float:
        """Share of the level's pixels that are tissue, in [0, 1]."""
        return float(self.pixels.mean())


def find_tissue(slide: OpenSlide) -> TissueMask:
    """Find tissue on the level closest to `TISSUE_MPP`.

    Args:
        slide: Open slide with an mpp.

    Returns:
        The tissue mask.

    Raises:
        InvalidInvocation: The closest level exceeds `MAX_PIXELS`, or it has too
            few grey values to threshold (a blank slide).
    """
    level = slide.closest_level(TISSUE_MPP)
    width, height = slide.level_dimensions[level]
    if width * height > MAX_PIXELS:
        raise InvalidInvocation(
            f"the slide has no level near {TISSUE_MPP} µm/px for tissue detection "
            f"(closest is {width}x{height} px); run with skip_background=false"
        )
    rgba = np.asarray(slide.read_region((0, 0), level, (width, height)))
    image = (
        pyvips.Image.new_from_array(rgba)
        .copy(interpretation="srgb")
        .flatten(background=255)
    )
    try:
        mask = tissue_mask(image, slide.slide_resolution(level))
    except ValueError as error:  # multi-Otsu cannot split so few grey values
        raise InvalidInvocation(f"tissue detection failed: {error}") from error
    return TissueMask(mask.numpy() > 0)
