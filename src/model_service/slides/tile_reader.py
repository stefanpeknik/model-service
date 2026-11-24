"""Read model tiles at the slide level closest to a model's resolution.

With a mask, a tile is read only when its mask window contains a nonzero pixel;
with a tissue mask, only when more than `min_tissue_fraction` of its window is
tissue (any tissue at 0). Reads are synchronous
and run in the pipeline runner's read threads; x/y are coordinates at the selected slide
level.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlideError
from ratiopath.openslide import OpenSlide

from model_service.contracts.failures import InvalidInvocation
from model_service.slides.tissue import find_tissue


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

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


def select_mask_level(mask: OpenSlide, target_mpp: float | tuple[float, float]) -> int:
    """Use the mask's resolution when it has one; otherwise its finest level.

    Args:
        mask: Open mask slide.
        target_mpp: Resolution to match.

    Returns:
        The mask level to read.
    """
    if (
        PROPERTY_NAME_MPP_X in mask.properties
        and PROPERTY_NAME_MPP_Y in mask.properties
    ):
        return mask.closest_level(target_mpp)
    return 0


class TileReader:
    """Reads model-sized tiles from one slide level, skipping masked-out and background tiles."""

    def __init__(
        self,
        slide: OpenSlide,
        mask: OpenSlide | None,
        mpp: float,
        tissue: TissueMask | None = None,
        min_tissue_fraction: float = 0.0,
    ) -> None:
        """Pick the slide level for the model and scale mask and tissue to it.

        Args:
            slide: Open slide to read tiles from.
            mask: Open region-of-interest mask, if any.
            mpp: Model resolution; the closest slide level is read.
            tissue: Tissue mask for skipping background, if any.
            min_tissue_fraction: Share of a tile's tissue window that must be
                tissue; 0 reads any tile with some tissue.
        """
        self.slide = slide
        self.level = slide.closest_level(mpp)
        self.tissue = tissue
        self._min_tissue_fraction = min_tissue_fraction
        self._mask = mask
        if tissue is not None:
            width, height = slide.level_dimensions[self.level]
            self._tissue_scale_x = tissue.pixels.shape[1] / width
            self._tissue_scale_y = tissue.pixels.shape[0] / height
        if mask is not None:
            self._mask_level = select_mask_level(mask, mpp)
            width, height = slide.level_dimensions[self.level]
            mask_width, mask_height = mask.level_dimensions[self._mask_level]
            self._scale_x = mask_width / width
            self._scale_y = mask_height / height

    @classmethod
    @contextmanager
    def open(
        cls,
        slide: OpenSlide,
        mask_path: Path | None,
        mpp: float,
        *,
        skip_background: bool = False,
        min_tissue_fraction: float = 0.0,
    ) -> Iterator[TileReader]:
        """Open a reader, with the mask file and tissue detection when requested.

        Args:
            slide: Open slide to read tiles from.
            mask_path: Region-of-interest mask file, if any.
            mpp: Model resolution.
            skip_background: Detect tissue and skip tiles without it.
            min_tissue_fraction: Share of a tile's window that must be tissue.

        Yields:
            The reader; the mask is closed on exit.

        Raises:
            InvalidInvocation: The mask is not a readable slide, or the slide
                has no level coarse enough for tissue detection.
        """
        tissue = find_tissue(slide) if skip_background else None
        if mask_path is None:
            yield cls(slide, None, mpp, tissue, min_tissue_fraction)
            return
        with open_slide(mask_path, "region_of_interest") as mask:
            yield cls(slide, mask, mpp, tissue, min_tissue_fraction)

    def read(self, x: int, y: int, tile_size: int) -> NDArray[np.uint8] | None:
        """Read one tile unless the tissue or mask window excludes it.

        Args:
            x: Tile origin x at the selected level.
            y: Tile origin y at the selected level.
            tile_size: Tile width and height in pixels.

        Returns:
            A CHW uint8 tile, or None when the tissue or mask window is empty.
        """
        # The tissue check is an array lookup; the mask check reads the mask file.
        if self.tissue is not None and not self._has_tissue(
            self.tissue, x, y, tile_size
        ):
            return None
        if self._mask is not None and not self._mask_selects(
            self._mask, x, y, tile_size
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

    def _mask_selects(self, mask: OpenSlide, x: int, y: int, tile_size: int) -> bool:
        """Check whether the tile's window in the mask has a nonzero pixel.

        Args:
            mask: The region-of-interest mask.
            x: Tile origin x at the selected level.
            y: Tile origin y at the selected level.
            tile_size: Tile width and height in pixels.

        Returns:
            True when the window contains a nonzero mask pixel.
        """
        left, top, width, height = _window(
            x, y, tile_size, self._scale_x, self._scale_y
        )
        window = mask.read_region_relative(
            (left, top), self._mask_level, (width, height)
        )
        return bool(np.asarray(window.convert("L")).any())


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
