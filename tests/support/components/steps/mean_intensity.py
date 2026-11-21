from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import Field

from model_service.components import Col, Options, PerRun, Problem, Step


if TYPE_CHECKING:
    from model_service.components import Batch


class MeanIntensityOptions(Options):
    channels: int = Field(1, ge=1, le=3, description="Colour channels to score.")
    scale: PerRun[float] = Field(1.0, gt=0, description="Multiplies every score.")


class MeanIntensity(Step):
    """Test step: the mean of each colour channel of a tile, scaled."""

    name = "mean_intensity"
    Options = MeanIntensityOptions
    reads: ClassVar[dict[str, Col]] = {
        "tile": Col("uint8", (3, "model.input.tile_size", "model.input.tile_size"))
    }
    adds: ClassVar[dict[str, Col]] = {"scores": Col("float32", ("channels", 1, 1))}

    def __call__(self, batch: Batch) -> Batch:
        """Add `scores`, one value per channel and tile."""
        assert isinstance(self.options, MeanIntensityOptions)
        tiles = batch["tile"][:, : self.options.channels].astype(np.float32)
        scores = tiles.mean(axis=(2, 3), keepdims=True) / 255 * self.options.scale
        return batch | {"scores": scores.astype(np.float32)}

    @classmethod
    def hint(cls, problem: Problem) -> str | None:
        """Explain that a tile has at most three channels."""
        if problem.detail.get("option") == "channels":
            return "tiles are RGB, so pick 1, 2 or 3 channels"
        return None
