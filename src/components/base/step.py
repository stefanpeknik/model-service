"""Step: maps a batch to a batch, row for row."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from components.base.component import Component
from components.base.slots import POST, PRE, STEPS


if TYPE_CHECKING:
    from components.base.columns import Batch
    from components.base.slots import Slot


class Step(Component):
    """Transforms data, row for row.

    The same step can be a served model's `pre:` or `post:` or an analysis
    step; the catalog file decides where it runs.
    """

    kind: ClassVar[str] = "step"
    slots: ClassVar[tuple[Slot, ...]] = (PRE, POST, STEPS)

    @abstractmethod
    def __call__(self, batch: Batch) -> Batch:
        """Return the batch with this step's columns added.

        Args:
            batch: Columns keyed by name; row i of each belongs to item i.

        Returns:
            A batch with the same number of rows and the declared `adds`.
        """
