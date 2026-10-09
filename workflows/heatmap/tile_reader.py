"""Read channel-first slide tiles only where the corresponding region mask contains selected pixels.

This synchronous helper runs in the processor's read thread pool; x/y coordinates belong
to the selected slide level.
"""

from math import ceil, floor

import numpy as np
from numpy.typing import NDArray
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y
from ratiopath.openslide import OpenSlide


def select_mask_level(mask: OpenSlide, target_mpp: float | tuple[float, float]) -> int:
    """Use mask resolution when known; otherwise preserve its finest pixel selection.

    Args:
        mask: Opened TIFF mask, which need not contain physical-resolution metadata.
        target_mpp: Selected slide resolution in micrometers per pixel.

    Returns:
        int: Closest physical-resolution level, or level zero when metadata is absent.
    """
    if (
        PROPERTY_NAME_MPP_X in mask.properties
        and PROPERTY_NAME_MPP_Y in mask.properties
    ):
        return mask.closest_level(target_mpp)
    return 0


def fetch_tissue_tile(
    slide: OpenSlide,
    tissue_slide: OpenSlide,
    x: int,
    y: int,
    level: int,
    scale_x: float,
    scale_y: float,
    tissue_level: int,
    tile_size: int,
) -> NDArray[np.uint8] | None:
    """Cover the tile window in mask coordinates and return None for an empty mask.

    Round mask edges outward so downsampled masks retain every intersecting pixel.

    Args:
        slide: Opened whole-slide image reader at the selected pyramid level.
        tissue_slide: Opened region-mask slide reader.
        x: Horizontal tile origin in selected slide-level pixels.
        y: Vertical tile origin in selected slide-level pixels.
        level: Selected pyramid level index.
        scale_x: Mask-to-slide width ratio at the selected pyramid levels.
        scale_y: Mask-to-slide height ratio at the selected pyramid levels.
        tissue_level: Selected mask pyramid level.
        tile_size: Input or provider publication tile extent in pixels.

    Returns:
        NDArray[np.uint8] | None: CHW uint8 slide pixels with shape channels by
            tile_size by tile_size, or None when the mask window contains no selected
            pixels.
    """
    mask_x, mask_y = floor(x * scale_x), floor(y * scale_y)
    mask_width = ceil((x + tile_size) * scale_x) - mask_x
    mask_height = ceil((y + tile_size) * scale_y) - mask_y
    tile = tissue_slide.read_region_relative(
        (mask_x, mask_y),
        tissue_level,
        (mask_width, mask_height),
    )
    if not np.asarray(tile.convert("L")).any():
        return None
    tile = slide.read_tile(
        x=x,
        y=y,
        extent_x=tile_size,
        extent_y=tile_size,
        level=level,
    )
    return tile.transpose(2, 0, 1)
