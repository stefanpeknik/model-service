from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from components import Col, Opened, Options, PerRun, Source


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from components import Batch


class GridOptions(Options):
    tile_size: int = Field(8, ge=1, description="Tile edge in pixels at level 0.")
    region_of_interest: Literal["required", "optional", "ignored"] = Field(
        "ignored", description="Whether a run must, may or may not give a ROI."
    )
    skip_background: PerRun[bool] = Field(
        False, description="Drop tiles whose pixels are all white."
    )


class Grid(Source):
    """Test source: non-overlapping tiles of a slide's level 0."""

    name = "grid"
    Options = GridOptions
    adds: ClassVar[dict[str, Col]] = {
        "tile": Col("uint8", (3, "tile_size", "tile_size")),
        "x": Col("int64"),
        "y": Col("int64"),
    }
    frame: ClassVar[dict[str, Col]] = {
        "extent_x": Col("int64"),
        "extent_y": Col("int64"),
        "tile_extent_x": Col("int64"),
        "tile_extent_y": Col("int64"),
    }

    def inputs(self) -> dict[str, bool]:
        """A slide, and a ROI unless the options ignore it."""
        assert isinstance(self.options, GridOptions)
        roi = self.options.region_of_interest
        return {"slide": True} | (
            {} if roi == "ignored" else {"roi": roi == "required"}
        )

    @contextmanager
    def open(self, inputs: dict[str, Path], run: Any) -> Iterator[Opened]:
        """Read level 0 with pyvips and cut it into tiles."""
        import pyvips

        assert isinstance(self.options, GridOptions)
        size = self.options.tile_size
        image = pyvips.Image.new_from_file(str(inputs["slide"]))
        pixels = np.asarray(image.extract_band(0, n=3).numpy())
        keys = [
            (x, y)
            for y in range(0, image.height - size + 1, size)
            for x in range(0, image.width - size + 1, size)
        ]
        skip = self.options.skip_background

        def load(chosen: Sequence[tuple[int, int]]) -> Batch:
            tiles = [
                pixels[y : y + size, x : x + size].transpose(2, 0, 1) for x, y in chosen
            ]
            kept = [i for i, t in enumerate(tiles) if not (skip and t.min() == 255)]
            return {
                "tile": np.stack([tiles[i] for i in kept]).astype(np.uint8)
                if kept
                else np.empty((0, 3, size, size), np.uint8),
                "x": np.array([chosen[i][0] for i in kept], np.int64),
                "y": np.array([chosen[i][1] for i in kept], np.int64),
            }

        frame = {
            "extent_x": image.width,
            "extent_y": image.height,
            "tile_extent_x": size,
            "tile_extent_y": size,
        }
        yield Opened(frame=frame, keys=keys, load=load)
