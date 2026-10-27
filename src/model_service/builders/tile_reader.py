"""Read model tiles at the slide level closest to a model's resolution.

With a mask, a tile is read only when its mask window contains a nonzero pixel;
with a tissue mask, only when its window contains tissue. Reads are synchronous
and run in the builder's thread pool; x/y are coordinates at the selected slide
level.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import ceil, floor
from typing import TYPE_CHECKING

import numpy as np
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlideError
from ratiopath.openslide import OpenSlide

from model_service.builders.contracts import InvalidInvocation
from model_service.builders.tissue import find_tissue


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from model_service.builders.tissue import TissueMask


@contextmanager
def open_slide(path: Path, name: str) -> Iterator[OpenSlide]:
    """Open an input slide; an unreadable file is the caller's error, not a retry."""
    try:
        slide = OpenSlide(str(path))
    except OpenSlideError as error:
        raise InvalidInvocation(f"input {name!r} is not a readable slide") from error
    with slide:
        yield slide


def select_mask_level(mask: OpenSlide, target_mpp: float | tuple[float, float]) -> int:
    """Use the mask's resolution when it has one; otherwise its finest level."""
    if (
        PROPERTY_NAME_MPP_X in mask.properties
        and PROPERTY_NAME_MPP_Y in mask.properties
    ):
        return mask.closest_level(target_mpp)
    return 0


class TileReader:
    def __init__(
        self,
        slide: OpenSlide,
        mask: OpenSlide | None,
        mpp: float,
        tissue: TissueMask | None = None,
    ) -> None:
        self.slide = slide
        self.level = slide.closest_level(mpp)
        self.tissue = tissue
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
    ) -> Iterator[TileReader]:
        tissue = find_tissue(slide) if skip_background else None
        if mask_path is None:
            yield cls(slide, None, mpp, tissue)
            return
        with open_slide(mask_path, "region_of_interest") as mask:
            yield cls(slide, mask, mpp, tissue)

    def read(self, x: int, y: int, tile_size: int) -> NDArray[np.uint8] | None:
        """Return a CHW uint8 tile, or None when the mask window is empty."""
        # The tissue check is an array lookup; the mask check reads the mask file.
        if self.tissue is not None and not self._has_tissue(x, y, tile_size):
            return None
        if self._mask is not None and not self._mask_selects(x, y, tile_size):
            return None
        tile = self.slide.read_tile(
            x=x, y=y, extent_x=tile_size, extent_y=tile_size, level=self.level
        )
        return tile.transpose(2, 0, 1)

    def _has_tissue(self, x: int, y: int, tile_size: int) -> bool:
        assert self.tissue is not None
        left, top = floor(x * self._tissue_scale_x), floor(y * self._tissue_scale_y)
        right = ceil((x + tile_size) * self._tissue_scale_x)
        bottom = ceil((y + tile_size) * self._tissue_scale_y)
        return bool(self.tissue.pixels[top:bottom, left:right].any())

    def _mask_selects(self, x: int, y: int, tile_size: int) -> bool:
        assert self._mask is not None
        # Round outward so a downsampled mask keeps every intersecting pixel.
        mask_x, mask_y = floor(x * self._scale_x), floor(y * self._scale_y)
        width = ceil((x + tile_size) * self._scale_x) - mask_x
        height = ceil((y + tile_size) * self._scale_y) - mask_y
        window = self._mask.read_region_relative(
            (mask_x, mask_y), self._mask_level, (width, height)
        )
        return bool(np.asarray(window.convert("L")).any())
