from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from model_service.components import Col, Sink


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.components import Batch


class ScoreTable(Sink):
    """Test sink: every row's scores in one .npz file."""

    name = "score_table"
    format = "test_score_table"
    reads: ClassVar[dict[str, Col]] = {"scores": Col("float32", (None, 1, 1))}
    frame_reads = frozenset({"extent_x", "extent_y"})
    example_options: ClassVar[dict[str, Any]] = {"output": "scores"}

    def open(self, frame: dict[str, Any], run: Any) -> None:
        """Remember where to write and the slide's size."""
        self.path: Path = run.work_dir / f"{self.options.output}.npz"
        self.extent = (frame["extent_x"], frame["extent_y"])
        self.parts: list[np.ndarray] = []

    def add(self, batch: Batch) -> None:
        """Keep the batch's scores."""
        self.parts.append(batch["scores"].reshape(len(batch["scores"]), -1))

    def close(self) -> Path:
        """Write all scores, possibly none."""
        scores = np.concatenate(self.parts) if self.parts else np.empty((0, 0))
        np.savez(self.path, scores=scores, extent=np.array(self.extent))
        return self.path
