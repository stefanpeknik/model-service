"""The raw scores behind a heatmap, kept beside its TIFF.

The TIFF is the scores upsampled to a slide level, so a score of 0 and a cell
the builder never scored look the same in it. This file keeps what the model
produced: one score per cell, which cells were scored, and how many pixels of
the level a cell covers. A provider that publishes scores, not pictures, reads
it instead of the TIFF.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


SCORE_GRID_SUFFIX = ".scores.npz"
# Cells of one pixel are a full-resolution mask, which is the picture itself.
MIN_CELL_SIZE = 2.0


@dataclass(frozen=True, slots=True)
class ScoreGrid:
    """Scores in 0..1 per cell, which cells were scored, and the cell's size."""

    values: NDArray[np.float32]
    scored: NDArray[np.bool_]
    # Pixels of the level the heatmap was built at that one cell covers.
    cell_size: float
    # Width and height of that level in pixels.
    level_size: tuple[int, int]


def score_grid_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + SCORE_GRID_SUFFIX)


def write_score_grid(artifact: Path, grid: ScoreGrid) -> None:
    """Write the grid beside `artifact`; it appears under its name once complete."""
    path = score_grid_path(artifact)
    temporary = path.with_name(f".{path.name}.partial.npz")
    np.savez_compressed(
        temporary,
        values=grid.values,
        scored=grid.scored,
        cell_size=np.float64(grid.cell_size),
        level_size=np.asarray(grid.level_size, dtype=np.int64),
    )
    os.replace(temporary, path)


def read_score_grid(artifact: Path) -> ScoreGrid | None:
    """The grid beside `artifact`, or None when the builder wrote none."""
    path = score_grid_path(artifact)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as stored:
        width, height = (int(value) for value in stored["level_size"])
        return ScoreGrid(
            values=stored["values"],
            scored=stored["scored"],
            cell_size=float(stored["cell_size"]),
            level_size=(width, height),
        )
