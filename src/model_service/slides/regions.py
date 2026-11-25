"""Where regions of interest meet a tile, with the team's geometry.

Ratiopath's `tile_annotations` cuts level-0 regions to a tile at the level
read, and rasterio's `geometry_mask`, as ratiopath's overlays use it, tells
which pixels of a map over the tile have their centre inside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray
    from shapely.geometry.base import BaseGeometry


def tile_part(
    regions: Sequence[BaseGeometry],
    x: int,
    y: int,
    size: tuple[int, int],
    downsample: float,
) -> BaseGeometry:
    """The part of the regions a tile covers.

    Args:
        regions: Regions in level-0 pixels.
        x: Tile origin x at the level read.
        y: Tile origin y at the level read.
        size: Tile width and height at the level read.
        downsample: The level's downsample from level 0.

    Returns:
        The covered part in the level's pixels; empty, or without area, when
        the tile only touches the regions.
    """
    import numpy as np
    from ratiopath.tiling.annotations import tile_annotations
    from shapely import box

    return next(
        tile_annotations(regions, box(0, 0, *size), [np.array([x, y])], downsample)
    )


def tile_pixels(
    part: BaseGeometry,
    x: int,
    y: int,
    size: tuple[int, int],
    shape: tuple[int, int],
) -> NDArray[np.bool_]:
    """Which pixels of a map over the tile have their centre in `part`.

    Args:
        part: A geometry in the level's pixels, e.g. from `tile_part`.
        x: Tile origin x at the level read.
        y: Tile origin y at the level read.
        size: Tile width and height at the level read.
        shape: The map's rows and columns.

    Returns:
        True for each pixel inside.
    """
    import rasterio.features
    from rasterio.transform import Affine

    rows, columns = shape
    outside = rasterio.features.geometry_mask(
        geometries=[part],
        out_shape=shape,
        transform=Affine.translation(x, y)
        * Affine.scale(size[0] / columns, size[1] / rows),
    )
    return ~outside
