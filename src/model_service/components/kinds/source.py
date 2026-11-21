"""Source: opens a run's inputs and yields items by key."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from model_service.components.component import Component
from model_service.components.slots import SOURCE


if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from pathlib import Path

    from model_service.components.columns import Batch, Col
    from model_service.components.slots import Slot


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


class Source(Component):
    """Produces an analysis's items from its inputs, e.g. tiles of a slide.

    Attributes:
        frame: The fields of the frame row the source gives.
    """

    kind: ClassVar[str] = "source"
    slots: ClassVar[tuple[Slot, ...]] = (SOURCE,)
    frame: ClassVar[dict[str, Col]] = {}

    def inputs(self) -> dict[str, bool]:
        """The inputs a run must or may give; they may follow from options.

        Returns:
            Input name to whether it is required.
        """
        return {}

    @abstractmethod
    def open(self, inputs: dict[str, Path], run: Any) -> AbstractContextManager[Opened]:
        """Open the inputs for one run.

        Args:
            inputs: Local paths of the run's inputs, keyed by input name.
            run: The run's context (served models, working directory).

        Returns:
            A context manager giving the open source and closing it after.
        """
