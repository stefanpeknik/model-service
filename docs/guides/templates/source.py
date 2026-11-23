"""The `my_component` source: <what items it makes from the inputs>.

Copy to `src/model_service/components/analysis/sources/<name>.py` and replace
`my_component`, `MyComponent` and the placeholders in <angle brackets>.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import Field

from model_service.components import Col, Opened, Options, Source


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from model_service.components import Batch, Run


class MyComponentOptions(Options):
    """Options of the `my_component` source.

    Attributes:
        tile_size: Tile edge in pixels.
    """

    tile_size: int = Field(64, ge=1, description="Tile edge in pixels.")


class MyComponent(Source):
    """<One sentence: which items it yields, from which inputs>.

    `source: {use: my_component}`. Its `inputs()` decide what a run gives.
    """

    name = "my_component"
    Options = MyComponentOptions
    options: MyComponentOptions
    adds: ClassVar[dict[str, Col]] = {
        "tile": Col("uint8", (3, "tile_size", "tile_size")),
        "tile_x": Col("int64"),
        "tile_y": Col("int64"),
    }
    frame: ClassVar[dict[str, Col]] = {
        "extent_x": Col("int64"),
        "extent_y": Col("int64"),
    }

    def inputs(self) -> dict[str, bool]:
        """The inputs a run gives.

        Returns:
            Input name to whether it is required.
        """
        return {"slide": True}

    @contextmanager
    def open(self, inputs: dict[str, Path], run: Run) -> Iterator[Opened]:
        """Open the slide and list its tiles.

        Args:
            inputs: Local paths by input name.
            run: The run; `run.model` is the last model step's served model.

        Yields:
            The frame, one key per tile and the function loading them.
        """
        import pyvips

        size = self.options.tile_size
        image = pyvips.Image.new_from_file(str(inputs["slide"]))
        pixels = np.asarray(image.extract_band(0, n=3).numpy())
        keys = [
            (x, y)
            for y in range(0, image.height - size + 1, size)
            for x in range(0, image.width - size + 1, size)
        ]

        def load(chosen: Sequence[tuple[int, int]]) -> Batch:
            tiles = [pixels[y : y + size, x : x + size] for x, y in chosen]
            return {
                "tile": np.stack(tiles).transpose(0, 3, 1, 2).astype(np.uint8)
                if tiles
                else np.empty((0, 3, size, size), np.uint8),
                "tile_x": np.array([x for x, _ in chosen], np.int64),
                "tile_y": np.array([y for _, y in chosen], np.int64),
            }

        yield Opened(
            frame={"extent_x": image.width, "extent_y": image.height},
            keys=keys,
            load=load,
        )
