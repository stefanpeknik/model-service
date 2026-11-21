from __future__ import annotations

from typing import TYPE_CHECKING, Any

from model_service.components import Commit, Publisher


if TYPE_CHECKING:
    from pathlib import Path


class _Noop:
    def commit(self) -> None:
        pass


class Text(Publisher):
    """Test publisher: shares its name with the `text` converter."""

    name = "text"
    reads_format = "value"
    ead_type = "string"

    def ead_fragment(self) -> dict[str, Any]:
        """Declare a string output."""
        return {"type": "string"}

    def prepare(self, artifact: Path, target: Any) -> Commit:
        """Nothing to prepare."""
        return _Noop()
