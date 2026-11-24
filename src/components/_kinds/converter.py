"""Input converter: turns one EMPAIA input type into a typed run input."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from components._kinds.component import Component
from components._kinds.slots import EMPAIA_INPUTS


if TYPE_CHECKING:
    from pathlib import Path

    from components._kinds.slots import Slot
    from model_service.contracts.json_values import JsonValue


@dataclass(frozen=True)
class EmpaiaInput:
    """One input of an EMPAIA app, as a converter sees it.

    Attributes:
        name: The EAD io key, the pipeline's input or setting name.
        slide: The io key of the app's `wsi` input, which other entries
            reference; None if the app has none.
        setting: For a standalone parameter, the JSON Schema of the setting
            (its `type` and `description`).
    """

    name: str
    slide: str | None = None
    setting: dict[str, Any] = field(default_factory=dict)


class InputConverter(Component):
    """Converts one EMPAIA input type (`wsi`, `rectangle`, ...) into a run input.

    The pipeline's `providers.empaia` declares each input's type, and that type
    picks the converter, so nothing is guessed from the data. A converter
    contributes the input's EAD io entry and materialises it for a run: a
    file in the attempt's workspace, or a value read at admission.

    Attributes:
        ead_type: The EAD input type it converts.
        gives_file: True if `materialize` writes a file the pipeline reads;
            False if it gives a value, read when the job is admitted.
    """

    kind: ClassVar[str] = "converter"
    folder: ClassVar[str] = "providers/empaia/converters"
    slots: ClassVar[tuple[Slot, ...]] = (EMPAIA_INPUTS,)
    ead_type: ClassVar[str]
    gives_file: ClassVar[bool] = True

    @abstractmethod
    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """The EAD `io` entry of the input.

        Args:
            item: The input.

        Returns:
            The entry, e.g. `{"type": "wsi"}`.
        """

    @abstractmethod
    async def materialize(
        self, gateway: Any, item: EmpaiaInput, directory: Path
    ) -> Path | JsonValue:
        """Fetch the job's input through the gateway's input reads.

        Args:
            gateway: The job's `EmpaiaGateway`.
            item: The input.
            directory: An empty directory of its own for a file.

        Returns:
            The file in `directory`, or the value.

        Raises:
            EmpaiaGatewayError: With a safe message, if the input cannot be read.
        """
