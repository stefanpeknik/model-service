from pathlib import Path
from typing import Any

from model_service.components import Batch, Sink


class StraySink(Sink):
    """Test sink placed in the steps folder."""

    name = "stray_sink"
    format = "none"

    def open(self, frame: dict[str, Any], run: Any) -> None:
        """Do nothing."""

    def add(self, batch: Batch) -> None:
        """Do nothing."""

    def close(self) -> Path:
        """Return no real file."""
        return Path()
