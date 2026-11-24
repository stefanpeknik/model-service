from __future__ import annotations

from typing import TYPE_CHECKING, Any

from components import EmpaiaInput, InputConverter


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.contracts.json_values import JsonValue


class Text(InputConverter):
    """Test converter: an EMPAIA string input, passed through."""

    name = "text"
    ead_type = "string"
    gives_file = False

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """Declare a string input."""
        return {"type": "string"}

    async def materialize(
        self, gateway: Any, item: EmpaiaInput, directory: Path
    ) -> Path | JsonValue:
        """Read the value as a string."""
        return str(await gateway.get_input_data(item.name))
