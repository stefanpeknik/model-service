"""A `{model: <id>}` step: one served model called on every tile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from components import Col
from model_service.catalog.served_models.fields import ScoresOutput


if TYPE_CHECKING:
    from model_service.catalog.served_models import ServedModel


@dataclass(frozen=True)
class ModelStep:
    """Calls a served model on each row's `tile`.

    The column it adds follows the model's `output.kind`: `scores` as
    float32 `(channels, size, size)`, or `features` as float32 `(dim,)` for one
    summary and `(summaries, dim)` for several.

    Attributes:
        model: The served model.
    """

    model: ServedModel

    @property
    def label(self) -> str:
        """How errors name the step."""
        return f"model step `{self.model.id}`"

    @property
    def column(self) -> str:
        """The column the step adds: `scores` or `features`."""
        return "scores" if isinstance(self.model.output, ScoresOutput) else "features"

    def columns(self) -> tuple[dict[str, Col], dict[str, Col]]:
        """The column read and the column added, with the model's dimensions.

        Returns:
            `tile` read; `scores` or `features` added.
        """
        origin = f"model {self.model.id}"
        size = self.model.input.tile_size
        reads = {"tile": Col("uint8", (3, size, size), origin)}
        output = self.model.output
        if isinstance(output, ScoresOutput):
            shape: tuple[int | None, ...] = (output.channels, output.size, output.size)
        else:
            count = len(output.summaries)
            shape = (None,) if count == 1 else (count, None)
        return reads, {self.column: Col("float32", shape, origin)}
