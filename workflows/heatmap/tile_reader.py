"""Read channel-first slide tiles only where the corresponding region mask contains selected pixels.

This synchronous helper runs in the processor's read thread pool; x/y coordinates belong
to the selected slide level.
"""

import numpy as np
from numpy.typing import NDArray
from ratiopath.openslide import OpenSlide


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
    """Map the tile window into mask coordinates and return None if its grayscale mask is empty; otherwise read the slide tile and transpose HWC pixels to CHW.

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
    tile = tissue_slide.read_region_relative(
        (int(x * scale_x), int(y * scale_y)),
        tissue_level,
        (int(tile_size * scale_x), int(tile_size * scale_y)),
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
