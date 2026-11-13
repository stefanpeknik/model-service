"""How builders appear in EMPAIA: the EAD `io` entry of their outputs.

A builder runs EMPAIA jobs only if it is listed in BUILDER_IO, with an entry for
each of its outputs; its `type` picks the output's publisher. The builders
themselves know nothing of EMPAIA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.contracts.json_values import JsonValue


type IoEntry = Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class BuilderIo:
    outputs: Mapping[str, IoEntry]

    def output_type(self, name: str) -> str | None:
        """Return the EMPAIA type an output's io entry declares.

        Args:
            name: The builder output name.

        Returns:
            The entry's `type`, or None if the output has no entry or no string type.
        """
        output_type = self.outputs.get(name, {}).get("type")
        return output_type if isinstance(output_type, str) else None


BUILDER_IO: dict[str, BuilderIo] = {
    "heatmap": BuilderIo(
        outputs={"heatmap": {"type": "continuous_pixelmap", "reference": "slide"}},
    ),
}
