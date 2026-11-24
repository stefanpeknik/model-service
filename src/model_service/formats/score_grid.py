"""`score_grid` v1: scores, one per cell of a grid laid over a slide level.

Media type `application/vnd.rationai.score-grid+npz; version=1`. A NumPy
`.npz` archive holding what the model produced, at the resolution it produced
it:

- `values`: float32 scores in 0..1, shaped (channels, rows, columns);
- `scored`: bool (rows, columns), whether any tile covered the cell, so a
  cell the model gave 0 differs from one it never scored;
- `level`: int64, the slide level the scores were computed at;
- `level_size`: int64 (2,), that level's width and height in pixels;
- `cell_size`: float64, pixels of that level one cell covers along each axis;
- `mpp`: float64 (2,), microns per pixel of that level, x and y.

Cell (r, c) covers the level's pixels from (c, r) * cell_size; the grid may
reach past the level's edge. Reading it needs only NumPy (`numpy.load`), and an
image of the slide's size is the grid scaled by `cell_size`. Several channels
are allowed; a consumer that needs one (an EMPAIA continuous pixelmap) checks
it itself. Written by the `score_grid` sink.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from model_service.formats.media_type import Format, UnsupportedFormat, require


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


FORMAT = Format(
    name="score_grid",
    version=1,
    extension="npz",
    type_name="score-grid",
    container="npz",
)

__all__ = [
    "FORMAT",
    "ScoreGrid",
    "UnsupportedFormat",
    "read",
    "write",
]


@dataclass(frozen=True, slots=True)
class ScoreGrid:
    """A score grid, as described in the module docstring.

    Attributes:
        values: float32 scores in 0..1, shaped (channels, rows, columns).
        scored: bool (rows, columns), whether any tile covered the cell.
        level: The slide level the scores were computed at.
        level_size: That level's width and height in pixels.
        cell_size: Pixels of that level one cell covers along each axis.
        mpp: Microns per pixel of that level, x and y.
    """

    values: NDArray[np.float32]
    scored: NDArray[np.bool_]
    level: int
    level_size: tuple[int, int]
    cell_size: float
    mpp: tuple[float, float]


def write(path: Path, grid: ScoreGrid) -> None:
    """Write a score grid as a compressed `.npz` archive.

    Args:
        path: File to write; NumPy keeps the name as given.
        grid: Scores and their placement on the slide level.
    """
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


def read(path: Path, media_type: str = FORMAT.media_type) -> ScoreGrid:
    """Read a score grid.

    Args:
        path: `.npz` file to read; pickled arrays are refused.
        media_type: The media type the file was stored with.

    Returns:
        The score grid.

    Raises:
        UnsupportedFormat: For a media type or version this reader doesn't know.
    """
    require(media_type, FORMAT.known)
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
