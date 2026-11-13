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
