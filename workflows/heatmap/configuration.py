"""Share score-map tiling and TIFF storage settings without selecting a workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from workflow_execution.contracts import JsonValue


class ScoreMapConfiguration(BaseModel):
    """Describe spatial reconstruction independently of output score interpretation.

    Attributes:
        output_path: Relative artifact destination supplied by the concrete workflow.
        stride_fraction: Input tile step divided by tile size, greater than zero to one.
        output_tile_height: TIFF storage tile height, in pixels.
        output_tile_width: TIFF storage tile width, in pixels.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    output_path: str = Field(min_length=1)
    stride_fraction: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    output_tile_height: int = Field(default=256, gt=0)
    output_tile_width: int = Field(default=256, gt=0)

    def as_values(self) -> dict[str, JsonValue]:
        """Return JSON-compatible defaults for catalog resolution.

        Returns:
            dict[str, JsonValue]: All declared settings including the concrete output path.
        """
        return self.model_dump()
