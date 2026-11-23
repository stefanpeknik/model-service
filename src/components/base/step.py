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

    def skipped(self, batch: Batch) -> Batch:
        """The batch a caller gets when it skips this step.

        The raw-inference route's `x-pool-tokens: false` skips a served
        model's `post:` steps. By default the batch passes unchanged; a
        pooling step may still drop what such callers never got, as
        `cls_mean_pool` drops register tokens.

        Args:
            batch: Columns keyed by name.

        Returns:
            The batch as a caller skipping this step sees it.
        """
        return batch
