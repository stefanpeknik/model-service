from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from components.base import Col, Sink


if TYPE_CHECKING:
    from pathlib import Path

    from components.base import Batch


class KeyLog(Sink):
    """Test sink that needs rows in key order: the tile positions as they came."""

    name = "key_log"
    format = "test_key_log"
    ordered = True
    reads: ClassVar[dict[str, Col]] = {"tile_x": Col("int64"), "tile_y": Col("int64")}
    example_options: ClassVar[dict[str, Any]] = {"output": "keys"}

    def open(self, frame: dict[str, Any], run: Any) -> None:
        """Start an empty log."""
        self.path: Path = run.work_dir / f"{self.options.output}.npy"
        self.rows: list[tuple[int, int]] = []

    def add(self, batch: Batch) -> None:
        """Log each row's (x, y)."""
        self.rows += zip(
            batch["tile_x"].tolist(), batch["tile_y"].tolist(), strict=True
        )

    def close(self) -> Path:
        """Write the log as an (n, 2) array."""
        np.save(self.path, np.array(self.rows, np.int64).reshape(-1, 2))
        return self.path
