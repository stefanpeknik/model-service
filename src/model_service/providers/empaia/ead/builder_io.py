"""How builders appear in EMPAIA: the EAD `io` entry of their inputs and outputs.

A builder is offered in EMPAIA only if it is listed in BUILDER_IO, with an entry for
each of its outputs and for each input an app takes; a reference names
another input. The builders themselves know nothing of EMPAIA.
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
    inputs: Mapping[str, IoEntry]
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
        inputs={
            "slide": {"type": "wsi"},
            "region_of_interest": {
                "type": "rectangle",
                "reference": "slide",
                "classes": ["org.empaia.global.v1.classes.roi"],
            },
        },
        outputs={"heatmap": {"type": "continuous_pixelmap", "reference": "slide"}},
    ),
}


# Registered builders deliberately not offered in EMPAIA, and why. Every
# builder is in BUILDER_IO or here; tests/providers/test_empaia_apps.py fails
# otherwise, so a new builder cannot skip the decision.
NOT_OFFERED: dict[str, str] = {}
