"""Source: opens a run's inputs and yields items by key."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from components.base.component import Component
from components.base.options import Options
from components.base.slots import SOURCE


if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from pathlib import Path

    from components.base.columns import Batch, Col
    from components.base.run import Run
    from components.base.slots import Slot


@dataclass
class Opened:
    """An open source.

    Attributes:
        frame: The one row describing the whole input (e.g. slide extents,
            level, stride), which sinks read through `frame_reads`.
        keys: One key per item, in run order.
        load: Loads the items of some keys; it may return fewer rows than keys
            when it drops items (e.g. background tiles).
    """

    frame: dict[str, Any]
    keys: Sequence[Any]
    load: Callable[[Sequence[Any]], Batch]


class TileGeometryOptions(Options):
    """Options of a source that reads tiles at a model's geometry.

    A source whose `Options` derive from this takes `tile_size` and `mpp`
    from the analysis's first model step wherever the file leaves them out;
    the catalog fills them in when the analysis loads and refuses a value
    that differs from the model's.

    Attributes:
        tile_size: Tile edge in pixels.
        mpp: Microns per pixel to read at.
    """

    tile_size: int | None = Field(
        None,
        gt=0,
        description="Tile edge in pixels; the first model step's when left out.",
        strict=True,
    )
    mpp: float | None = Field(
        None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Microns per pixel to read at (the closest slide level); the first "
            "model step's when left out."
        ),
    )


class Source(Component):
    """Produces an analysis's items from its inputs, e.g. tiles of a slide.

    A source whose options derive from `TileGeometryOptions` takes
    `tile_size` and `mpp` from the analysis's first model step wherever the
    file leaves them out; the catalog fills them in when the analysis loads.

    Attributes:
        frame: The fields of the frame row the source gives.
    """

    kind: ClassVar[str] = "source"
    folder: ClassVar[str] = "sources"
    slots: ClassVar[tuple[Slot, ...]] = (SOURCE,)
    frame: ClassVar[dict[str, Col]] = {}

    def inputs(self) -> dict[str, bool]:
        """The inputs a run must or may give; they may follow from options.

        Returns:
            Input name to whether it is required.
        """
        return {}

    @abstractmethod
    def open(self, inputs: dict[str, Path], run: Run) -> AbstractContextManager[Opened]:
        """Open the inputs for one run.

        Args:
            inputs: Local paths of the run's inputs, keyed by input name.
            run: The run's `Run`.

        Returns:
            A context manager giving the open source and closing it after.
        """
