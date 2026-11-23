"""The `resize` step: scale each uint8 RGB tile so its shorter edge has a size."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from components.base import Step
from components.base.columns import Col
from components.base.options import Options


if TYPE_CHECKING:
    from components.base.columns import Batch


class ResizeOptions(Options):
    """Options of the `resize` step.

    Attributes:
        size: The shorter edge after resizing, in pixels.
        interpolation: The resampling filter.
    """

    size: int = Field(
        gt=0,
        description=(
            "The shorter edge after resizing, in pixels; the longer edge keeps "
            "the aspect ratio, so a square tile becomes `size` x `size`."
        ),
        strict=True,
    )
    interpolation: Literal["bilinear", "bicubic", "nearest"] = Field(
        "bilinear",
        description="The resampling filter: `bilinear`, `bicubic` or `nearest`.",
    )


class Resize(Step):
    """Resizes uint8 RGB tiles, as torchvision's `Resize` does to a PIL image.

    `pre: [{use: resize, size: 256, interpolation: bicubic}]` scales a 224 px
    tile to 256 px, as Prov-GigaPath's transform does. Pillow does the
    resampling, with its antialiasing, so the pixels match torchvision's bit
    for bit; Pillow must be installed where the step runs.
    """

    name = "resize"
    Options = ResizeOptions
    example_options: ClassVar[dict[str, Any]] = {
        "size": 256,
        "interpolation": "bicubic",
    }
    reads: ClassVar[dict[str, Col]] = {"tile": Col("uint8", (3, None, None))}
    adds: ClassVar[dict[str, Col]] = {"tile": Col("uint8", (3, None, None))}
    options: ResizeOptions

    def __call__(self, batch: Batch) -> Batch:
        """Resize every tile.

        Args:
            batch: `tile`, uint8 RGB `(rows, 3, height, width)`.

        Returns:
            The batch with the resized tiles in `tile`.
        """
        from PIL import Image

        tiles = batch["tile"]
        height, width = self._output_size(*tiles.shape[2:])
        if (height, width) == tiles.shape[2:]:
            return batch
        resample = getattr(Image.Resampling, self.options.interpolation.upper())
        resized = [
            np.asarray(
                Image.fromarray(np.ascontiguousarray(tile.transpose(1, 2, 0))).resize(
                    (width, height), resample
                )
            ).transpose(2, 0, 1)
            for tile in tiles
        ]
        return batch | {"tile": np.stack(resized)}

    def _output_size(self, height: int, width: int) -> tuple[int, int]:
        """The size torchvision gives for a shorter-edge resize.

        Args:
            height: The input's height.
            width: The input's width.

        Returns:
            The output's `(height, width)`.
        """
        short, long = sorted((height, width))
        new_long = int(self.options.size * long / short)
        if height <= width:
            return self.options.size, new_long
        return new_long, self.options.size
