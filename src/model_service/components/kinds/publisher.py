"""Publisher: carries a stored output into EMPAIA via prepare, then commit."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from model_service.components.component import Component
from model_service.components.slots import EMPAIA_OUTPUTS


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.components.slots import Slot


class Commit(Protocol):
    """A prepared publication: everything is checked, only the write remains."""

    def commit(self) -> None:
        """Write the output to EMPAIA."""


class Publisher(Component):
    """Publishes an artifact of one data format as one EMPAIA output type.

    Attributes:
        reads_format: The data format of the artifact it publishes.
        ead_type: The EAD type it emits, e.g. `continuous_pixelmap`.
    """

    kind: ClassVar[str] = "publisher"
    slots: ClassVar[tuple[Slot, ...]] = (EMPAIA_OUTPUTS,)
    reads_format: ClassVar[str]
    ead_type: ClassVar[str]

    @abstractmethod
    def ead_fragment(self) -> dict[str, Any]:
        """The EAD entry this output contributes to the app's description."""

    @abstractmethod
    def prepare(self, artifact: Path, target: Any) -> Commit:
        """Check and prepare everything before anything is written.

        Args:
            artifact: The stored output.
            target: Where to publish (e.g. the EMPAIA gateway of the job).

        Returns:
            The commit that performs the write.
        """
