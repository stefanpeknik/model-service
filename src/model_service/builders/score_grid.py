"""The heatmap builder's artifact: the model's scores, one per cell of a grid.

A NumPy `.npz` archive holding what the model produced, at the resolution it
produced it:

- `values`: float32 scores in 0..1, shaped (channels, rows, columns);
- `scored`: bool (rows, columns), whether any tile covered the cell, so a
  cell the model gave 0 differs from one it never scored;
- `level`: the slide level the scores were computed at;
- `level_size`: that level's width and height in pixels;
- `cell_size`: pixels of that level one cell covers along each axis;
- `mpp`: microns per pixel of that level, x and y.

Cell (r, c) covers the level's pixels from (c, r) * cell_size; the grid may
reach past the level's edge. Reading it needs only NumPy (`numpy.load`), and an
image of the slide's size is the grid scaled by `cell_size`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


MEDIA_TYPE = "application/x-npz"


@dataclass(frozen=True, slots=True)
class ScoreGrid:
    values: NDArray[np.float32]
    scored: NDArray[np.bool_]
    level: int
    level_size: tuple[int, int]
    cell_size: float
    mpp: tuple[float, float]


def write_score_grid(path: Path, grid: ScoreGrid) -> None:
    # A file object, so NumPy keeps the name as given.
    with path.open("wb") as file:
        np.savez_compressed(
            file,
            values=grid.values,
            scored=grid.scored,
            level=np.int64(grid.level),
            level_size=np.asarray(grid.level_size, dtype=np.int64),
            cell_size=np.float64(grid.cell_size),
            mpp=np.asarray(grid.mpp, dtype=np.float64),
        )


def read_score_grid(path: Path) -> ScoreGrid:
    with np.load(path, allow_pickle=False) as stored:
        width, height = (int(value) for value in stored["level_size"])
        mpp_x, mpp_y = (float(value) for value in stored["mpp"])
        return ScoreGrid(
            values=stored["values"],
            scored=stored["scored"],
            level=int(stored["level"]),
            level_size=(width, height),
            cell_size=float(stored["cell_size"]),
            mpp=(mpp_x, mpp_y),
        )
