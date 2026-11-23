"""The `score_grid` data format: the npz layout under a versioned media type."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from model_service.formats import score_grid


if TYPE_CHECKING:
    from pathlib import Path


def write_layout(path: Path, **changes: object) -> Path:
    """An npz with the keys and dtypes of score_grid v1."""
    arrays = {
        "values": np.array([[[0.25, 1.0], [0.0, 0.5]]], np.float32),
        "scored": np.array([[True, True], [False, True]]),
        "level": np.int64(1),
        "level_size": np.array([300, 200], np.int64),
        "cell_size": np.float64(128.0),
        "mpp": np.array([0.5, 0.52], np.float64),
    } | changes
    with path.open("wb") as file:
        np.savez_compressed(file, **{k: v for k, v in arrays.items() if v is not None})
    return path


def test_a_file_in_the_layout_reads_and_checks(tmp_path: Path) -> None:
    path = write_layout(tmp_path / "grid.npz")

    grid = score_grid.read(path)

    assert score_grid.check(path) == []
    assert grid.values.dtype == np.float32
    assert grid.values.shape == (1, 2, 2)
    assert grid.scored.tolist() == [[True, True], [False, True]]
    assert (grid.level, grid.level_size, grid.cell_size, grid.mpp) == (
        1,
        (300, 200),
        128.0,
        (0.5, 0.52),
    )


def test_what_write_writes_reads_back_unchanged(tmp_path: Path) -> None:
    original = score_grid.read(write_layout(tmp_path / "a.npz"))

    score_grid.write(tmp_path / "b.npz", original)
    copy = score_grid.read(tmp_path / "b.npz")

    assert copy.values.tobytes() == original.values.tobytes()
    assert copy.scored.tolist() == original.scored.tolist()
    assert (copy.level, copy.level_size, copy.cell_size, copy.mpp) == (
        original.level,
        original.level_size,
        original.cell_size,
        original.mpp,
    )


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"scored": None}, "missing `scored`"),
        ({"values": np.full((1, 2, 2), 1.5, np.float32)}, "outside 0..1"),
        ({"values": np.zeros((1, 2, 2), np.float64)}, "`values` is float64"),
        ({"scored": np.ones((3, 2), bool)}, "`scored` is (3, 2)"),
    ],
)
def test_check_names_what_breaks_the_layout(
    tmp_path: Path, changes: dict[str, object], problem: str
) -> None:
    path = write_layout(tmp_path / "grid.npz", **changes)

    problems = score_grid.check(path)

    assert any(problem in p for p in problems), problems


def test_a_reader_refuses_a_version_it_does_not_know(tmp_path: Path) -> None:
    path = write_layout(tmp_path / "grid.npz")
    newer = score_grid.MEDIA_TYPE.replace("version=1", "version=2")

    assert score_grid.MEDIA_TYPE == (
        "application/vnd.rationai.score-grid+npz; version=1"
    )
    with pytest.raises(score_grid.UnsupportedFormat, match="version=2"):
        score_grid.read(path, media_type=newer)
