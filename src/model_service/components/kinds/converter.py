"""Input converter: turns an EMPAIA input into a typed run input."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from model_service.components.component import Component
from model_service.components.slots import EMPAIA_INPUTS


if TYPE_CHECKING:
    from model_service.components.slots import Slot


class InputConverter(Component):
    """Converts one EMPAIA input type (`wsi`, `rectangle`, ...) into a run input.

    Attributes:
        ead_type: The EAD type the converter accepts.
    """

    kind: ClassVar[str] = "converter"
    slots: ClassVar[tuple[Slot, ...]] = (EMPAIA_INPUTS,)
    ead_type: ClassVar[str]

    @abstractmethod
    def ead_fragment(self) -> dict[str, Any]:
        """The EAD entry this input contributes to the app's description."""

    @abstractmethod
    def convert(self, value: Any, context: Any) -> Any:
        """Materialise the input for a run.

        Args:
            value: The input as the EMPAIA job gives it.
            context: Access to the job's inputs (e.g. the gateway).

        Returns:
            The typed input the analysis's source takes.
        """
