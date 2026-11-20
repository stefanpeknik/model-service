"""Find tissue on a slide's coarse level so builders can skip empty glass.

Detection is ratiopath's `tissue_mask` (grayscale, multi-Otsu, closing and
opening) on the level ratiopath's `read_tile` reads, with transparent regions
(e.g. in MIRAX slides) composited onto white. Multi-Otsu treats only the
brightest class as glass, so on a slide with little tissue and a large white
fill, real glass can come out as tissue: filtering then skips less, it never
drops tissue. This is a stopgap until QC publishes tissue masks.
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
        InvalidInvocation: The closest level exceeds `MAX_PIXELS`, or it has
            fewer than four grey values to threshold.
    """
    level = slide.closest_level(TISSUE_MPP)
    width, height = slide.level_dimensions[level]
    if width * height > MAX_PIXELS:
        raise InvalidInvocation(
            f"the slide has no level near {TISSUE_MPP} µm/px for tissue detection "
            f"(closest is {width}x{height} px); run with skip_background=false"
        )
    rgb = slide.read_tile(0, 0, width, height, level)
    image = pyvips.Image.new_from_array(rgb).copy(interpretation="srgb")
    try:
        mask = tissue_mask(image, slide.slide_resolution(level))
    except ValueError as error:  # multi-Otsu needs at least four grey values
        raise InvalidInvocation(
            "the slide is too uniform for tissue detection; "
            "run with skip_background=false"
        ) from error
    return TissueMask(mask.numpy() > 0)
