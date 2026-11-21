from __future__ import annotations

from typing import Any

from model_service.components import InputConverter


class Text(InputConverter):
    """Test converter: an EMPAIA string input, passed through."""

    name = "text"
    ead_type = "string"

    def ead_fragment(self) -> dict[str, Any]:
        """Declare a string input."""
        return {"type": "string"}

    def convert(self, value: Any, context: Any) -> Any:
        """Return the value as a string."""
        return str(value)
