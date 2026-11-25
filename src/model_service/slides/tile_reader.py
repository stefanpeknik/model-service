"""Read model tiles at the slide level closest to a model's resolution.

With regions, a tile is read only when it overlaps one of them by some area:
ratiopath's `tile_annotations` intersects the tile with the level-0 regions.
With a tissue mask, only when more than `min_tissue_fraction` of its window is
tissue (any tissue at 0). Reads are synchronous and run in the pipeline
runner's read threads; x/y are coordinates at the selected slide level.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from openslide import OpenSlideError
from ratiopath.openslide import OpenSlide

from model_service.contracts.failures import InvalidInvocation
from model_service.formats import regions as regions_format
from model_service.slides.tissue import find_tissue


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from model_service.formats.regions import Region
    from model_service.slides.tissue import TissueMask


@contextmanager
def open_slide(path: Path, name: str) -> Iterator[OpenSlide]:
    """Open an input slide; an unreadable file is the caller's error, not a retry.

    Args:
        path: Slide file.
        name: Input name used in the error message.

    Yields:
        The open slide, closed on exit.

    Raises:
        InvalidInvocation: OpenSlide cannot read the file.
    """
    try:
        slide = OpenSlide(str(path))
    except OpenSlideError as error:
        raise InvalidInvocation(f"input {name!r} is not a readable slide") from error
    with slide:
        yield slide


class TileReader:
    """Reads model-sized tiles from one slide level, skipping tiles outside the regions and background."""

    def __init__(
        self,
        slide: OpenSlide,
        regions: tuple[Region, ...] | None,
        mpp: float,
        tissue: TissueMask | None = None,
        min_tissue_fraction: float = 0.0,
    ) -> None:
        """Pick the slide level for the model and scale tissue to it.

        Args:
            slide: Open slide to read tiles from.
            regions: Regions of interest in level-0 pixels, if any.
            mpp: Model resolution; the closest slide level is read.
            tissue: Tissue mask for skipping background, if any.
            min_tissue_fraction: Share of a tile's tissue window that must be
                tissue; 0 reads any tile with some tissue.
        """
        self.slide = slide
        self.level = slide.closest_level(mpp)
        self.downsample = float(slide.level_downsamples[self.level])
        self.regions = regions
        self.tissue = tissue
        self._min_tissue_fraction = min_tissue_fraction
        if tissue is not None:
            width, height = slide.level_dimensions[self.level]
            self._tissue_scale_x = tissue.pixels.shape[1] / width
            self._tissue_scale_y = tissue.pixels.shape[0] / height

    @classmethod
    @contextmanager
    def open(
        cls,
        slide: OpenSlide,
        regions_path: Path | None,
        mpp: float,
        *,
        skip_background: bool = False,
        min_tissue_fraction: float = 0.0,
    ) -> Iterator[TileReader]:
        """Open a reader, with the regions file and tissue detection when requested.

        Args:
            slide: Open slide to read tiles from.
            regions_path: Regions-of-interest file in the `regions` format, if any.
            mpp: Model resolution.
            skip_background: Detect tissue and skip tiles without it.
            min_tissue_fraction: Share of a tile's window that must be tissue.

        Yields:
            The reader.

        Raises:
            InvalidInvocation: The regions file is not readable regions, or the
                slide has no level coarse enough for tissue detection.
        """
        regions = None if regions_path is None else read_regions(regions_path)
        tissue = find_tissue(slide) if skip_background else None
        yield cls(slide, regions, mpp, tissue, min_tissue_fraction)

    def read(self, x: int, y: int, tile_size: int) -> NDArray[np.uint8] | None:
        """Read one tile unless the tissue window or the regions exclude it.

        Args:
            x: Tile origin x at the selected level.
            y: Tile origin y at the selected level.
            tile_size: Tile width and height in pixels.

        Returns:
            A CHW uint8 tile, or None when the tile has no tissue or lies
            outside every region.
        """
        if self.tissue is not None and not self._has_tissue(
            self.tissue, x, y, tile_size
        ):
            return None
        if self.regions is not None and not self._in_regions(
            self.regions, x, y, tile_size
        ):
            return None
        tile = self.slide.read_tile(
            x=x, y=y, extent_x=tile_size, extent_y=tile_size, level=self.level
        )
        # Contiguous here, in the read thread, so Ray need not copy it on the event loop.
        return np.ascontiguousarray(tile.transpose(2, 0, 1))

    def _has_tissue(self, tissue: TissueMask, x: int, y: int, tile_size: int) -> bool:
        """Check whether enough of the tile's window in the tissue mask is tissue.

        Args:
            tissue: The slide's tissue mask.
            x: Tile origin x at the selected level.
            y: Tile origin y at the selected level.
            tile_size: Tile width and height in pixels.

        Returns:
            True when the window's tissue share exceeds the minimum.
        """
        left, top, width, height = _window(
            x, y, tile_size, self._tissue_scale_x, self._tissue_scale_y
        )
        window = tissue.pixels[top : top + height, left : left + width]
        return window.size > 0 and float(window.mean()) > self._min_tissue_fraction

    def _in_regions(
        self, regions: tuple[Region, ...], x: int, y: int, tile_size: int
    ) -> bool:
        """Check whether the tile overlaps a region by some area; touching is not enough.

        Args:
            regions: The regions, in level-0 pixels.
            x: Tile origin x at the selected level.
            y: Tile origin y at the selected level.
            tile_size: Tile width and height in pixels.

        Returns:
            True when the tile and the regions share some area.
        """
        from ratiopath.tiling.annotations import tile_annotations
        from shapely import box

        overlap = next(
            tile_annotations(
                [r.geometry for r in regions],
                box(0, 0, tile_size, tile_size),
                [np.array([x, y])],
                self.downsample,
            )
        )
        return overlap.area > 0


def read_regions(path: Path) -> tuple[Region, ...]:
    """Read a regions-of-interest file; a malformed one is the caller's error.

    Args:
        path: The file, in the `regions` format.

    Returns:
        The regions.

    Raises:
        InvalidInvocation: The file is not readable regions.
    """
    from shapely.errors import ShapelyError

    try:
        return regions_format.read(path)
    except (OSError, ValueError, TypeError, KeyError, ShapelyError) as error:
        raise InvalidInvocation(
            "input 'region_of_interest' is not a GeoJSON feature collection of polygons"
        ) from error


def _window(
    x: int, y: int, tile_size: int, scale_x: float, scale_y: float
) -> tuple[int, int, int, int]:
    """Scale a tile into a coarser image the way the original model-service tiling did.

    Origin and size are scaled and truncated separately, as in its
    `fetch_tissue_tile`.

    Args:
        x: Tile origin x at the selected level.
        y: Tile origin y at the selected level.
        tile_size: Tile width and height in pixels.
        scale_x: The image's width over the level's.
        scale_y: The image's height over the level's.

    Returns:
        The window's left, top, width and height in the image's pixels.
    """
    return (
        int(x * scale_x),
        int(y * scale_y),
        int(tile_size * scale_x),
        int(tile_size * scale_y),
    )
