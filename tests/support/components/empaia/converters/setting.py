from __future__ import annotations

from typing import TYPE_CHECKING, Any

from components.base import EmpaiaInput, InputConverter


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.contracts.json_values import JsonValue


TYPES = {"boolean": "bool", "integer": "integer", "number": "float", "string": "string"}


class Setting(InputConverter):
    """Test converter: offers a standalone parameter as an EMPAIA primitive."""

    name = "setting"
    ead_type = "primitive"
    gives_file = False

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """Declare the setting as the primitive of its JSON Schema type."""
        kind = TYPES.get(item.setting.get("type", ""))
        if kind is None:
            raise ValueError(f"setting `{item.name}` is not a bool, number or string")
        return {"type": kind, "description": item.setting["description"]}

    async def materialize(
        self, gateway: Any, item: EmpaiaInput, directory: Path
    ) -> Path | JsonValue:
        """Read the value the user gave."""
        data = await gateway.get_input_data(item.name)
        return data["value"]
