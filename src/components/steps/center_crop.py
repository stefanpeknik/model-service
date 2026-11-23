"""The `center_crop` step: the centre square of each tile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field

from components.base import Step
from components.base.columns import Col
from components.base.options import Options


if TYPE_CHECKING:
    from components.base.columns import Batch


class CenterCropOptions(Options):
    """Options of the `center_crop` step.

    Attributes:
        size: The edge of the square kept, in pixels.
    """

    size: int = Field(
        gt=0, description="The edge of the centre square kept, in pixels.", strict=True
    )


class CenterCrop(Step):
    """Keeps the centre square of each uint8 tile, as torchvision's `CenterCrop`.

    `pre: [{use: center_crop, size: 224}]` keeps the centre 224 px. An odd
    margin puts the extra pixel where torchvision does (it rounds half to
    even), and a tile smaller than `size` is padded with zeros first, also as
    torchvision does.
    """

    name = "center_crop"
    Options = CenterCropOptions
    example_options: ClassVar[dict[str, Any]] = {"size": 224}
    reads: ClassVar[dict[str, Col]] = {"tile": Col("uint8", (3, None, None))}
    adds: ClassVar[dict[str, Col]] = {"tile": Col("uint8", (3, "size", "size"))}
    options: CenterCropOptions

    def __call__(self, batch: Batch) -> Batch:
        """Crop every tile to its centre square.

        Args:
            batch: `tile`, uint8 `(rows, 3, height, width)`.

        Returns:
            The batch with `tile` as `(rows, 3, size, size)`.
        """
        size = self.options.size
        tiles = batch["tile"]
        height, width = tiles.shape[2:]
        if height < size or width < size:
            pad_h, pad_w = max(size - height, 0), max(size - width, 0)
            tiles = np.pad(
                tiles,
                (
                    (0, 0),
                    (0, 0),
                    (pad_h // 2, (pad_h + 1) // 2),
                    (pad_w // 2, (pad_w + 1) // 2),
                ),
            )
            height, width = tiles.shape[2:]
        top = round((height - size) / 2)
        left = round((width - size) / 2)
        return batch | {"tile": tiles[:, :, top : top + size, left : left + size]}
