"""The `embedding_table` data format: today's HDF5 layout under a versioned media type."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest

from model_service.formats import embedding_table


if TYPE_CHECKING:
    from pathlib import Path


TODAYS_ATTRS: dict[str, Any] = {
    "schema_version": 2,
    "model_id": "virchow2",
    "summary_names": ("global", "local_mean"),
    "coordinate_space": "slide_level",
    "slide_level": 1,
    "level_downsample": 2.0,
    "slide_width": 300,
    "slide_height": 200,
    "mpp_x": 0.5,
    "mpp_y": 0.52,
    "model_mpp": 0.5,
    "tile_size": 224,
    "stride": 224,
    "tile_count": 2,
}


def write_todays_layout(
    path: Path,
    *,
    attrs: dict[str, Any] | None = None,
    coordinates: Any = ((0, 0), (224, 0)),
    features: Any = None,
) -> Path:
    """An HDF5 file with the datasets and attributes today's embeddings builder writes."""
    if features is None:
        features = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    with h5py.File(path, "w") as file:
        for key, value in (TODAYS_ATTRS | (attrs or {})).items():
            if value is not None:
                file.attrs[key] = value
        if coordinates is not None:
            file.create_dataset("coordinates", data=np.asarray(coordinates, np.int64))
        file.create_dataset("features", data=features)
    return path


def test_a_file_in_todays_layout_reads_and_checks(tmp_path: Path) -> None:
    path = write_todays_layout(tmp_path / "table.h5")

    table = embedding_table.read(path)

    assert embedding_table.check(path) == []
    assert table.coordinates.tolist() == [[0, 0], [224, 0]]
    assert table.features.dtype == np.float32
    assert table.features.shape == (2, 2, 3)
    assert table.header == embedding_table.Header(
        model_id="virchow2",
        summary_names=("global", "local_mean"),
        slide_level=1,
        level_downsample=2.0,
        slide_width=300,
        slide_height=200,
        mpp_x=0.5,
        mpp_y=0.52,
        model_mpp=0.5,
        tile_size=224,
        stride=224,
    )


@pytest.mark.parametrize(
    ("summaries", "features"),
    [
        (("tile",), np.array([[0.5, 1.5], [2.5, 3.5]], np.float32)),
        (("global", "local_mean"), np.ones((2, 2, 4), np.float32)),
        (("tile",), np.empty((0, 0), np.float32)),
        (("global", "local_mean"), np.empty((0, 2, 0), np.float32)),
    ],
    ids=["one summary", "two summaries", "empty, one summary", "empty, two"],
)
def test_what_write_writes_reads_back_and_checks(
    tmp_path: Path, summaries: tuple[str, ...], features: np.ndarray
) -> None:
    original = embedding_table.read(write_todays_layout(tmp_path / "a.h5"))
    header = replace(original.header, summary_names=summaries)
    coordinates = np.array([[0, 0], [224, 0]][: len(features)], np.int64).reshape(-1, 2)
    table = embedding_table.EmbeddingTable(header, coordinates, features)

    embedding_table.write(tmp_path / "b.h5", table)
    copy = embedding_table.read(tmp_path / "b.h5")

    assert embedding_table.check(tmp_path / "b.h5") == []
    assert copy.header == header
    assert copy.coordinates.tolist() == coordinates.tolist()
    assert copy.features.shape == features.shape
    assert copy.features.tobytes() == features.tobytes()
    with h5py.File(tmp_path / "b.h5") as file:
        assert file.attrs["tile_count"] == len(features)


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"attrs": {"model_id": None}}, "missing attribute `model_id`"),
        ({"attrs": {"schema_version": 1}}, "`schema_version` is 1, not 2"),
        ({"attrs": {"tile_count": 3}}, "`tile_count` is 3"),
        ({"coordinates": None}, "missing `coordinates`"),
        ({"features": np.ones((2, 2, 3), np.float64)}, "`features` is float64"),
        ({"features": np.ones((2, 3), np.float32)}, "one row per summary"),
        ({"features": np.ones((3, 2, 3), np.float32)}, "3 rows of features"),
        ({"features": np.full((2, 2, 3), np.nan, np.float32)}, "not finite"),
    ],
)
def test_check_names_what_breaks_the_layout(
    tmp_path: Path, change: dict[str, Any], problem: str
) -> None:
    path = write_todays_layout(tmp_path / "table.h5", **change)

    problems = embedding_table.check(path)

    assert any(problem in p for p in problems), problems


def test_check_refuses_a_file_that_is_not_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "table.h5"
    path.write_bytes(b"not hdf5")

    assert embedding_table.check(path)[0].startswith("not an HDF5 file")


def test_a_reader_takes_todays_media_type_and_refuses_an_unknown_version(
    tmp_path: Path,
) -> None:
    path = write_todays_layout(tmp_path / "table.h5")
    newer = embedding_table.MEDIA_TYPE.replace("version=2", "version=3")

    assert embedding_table.MEDIA_TYPE == (
        "application/vnd.rationai.embedding-table+hdf5; version=2"
    )
    legacy = embedding_table.read(path, media_type="application/x-hdf5")

    assert legacy.features.shape == (2, 2, 3)
    with pytest.raises(embedding_table.UnsupportedFormat, match="version=3"):
        embedding_table.read(path, media_type=newer)
