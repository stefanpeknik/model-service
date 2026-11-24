"""The `embedding_table` data format: the HDF5 layout under a versioned media type."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest

from model_service.formats import embedding_table


if TYPE_CHECKING:
    from pathlib import Path


LAYOUT_ATTRS: dict[str, Any] = {
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


def write_layout(
    path: Path,
    *,
    attrs: dict[str, Any] | None = None,
    coordinates: Any = ((0, 0), (224, 0)),
    features: Any = None,
) -> Path:
    """An HDF5 file with the datasets and attributes of embedding_table v2."""
    if features is None:
        features = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    with h5py.File(path, "w") as file:
        for key, value in (LAYOUT_ATTRS | (attrs or {})).items():
            if value is not None:
                file.attrs[key] = value
        if coordinates is not None:
            file.create_dataset("coordinates", data=np.asarray(coordinates, np.int64))
        file.create_dataset("features", data=features)
    return path


def test_a_file_in_the_layout_reads(tmp_path: Path) -> None:
    path = write_layout(tmp_path / "table.h5")

    table = embedding_table.read(path)

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
def test_what_the_table_writer_writes_reads_back(
    tmp_path: Path, summaries: tuple[str, ...], features: np.ndarray
) -> None:
    original = embedding_table.read(write_layout(tmp_path / "a.h5"))
    header = replace(original.header, summary_names=summaries)
    coordinates = np.array([[0, 0], [224, 0]][: len(features)], np.int64).reshape(-1, 2)

    with embedding_table.TableWriter(tmp_path / "b.h5", header) as writer:
        if len(coordinates):
            writer.append(coordinates, features)
        writer.close()
    copy = embedding_table.read(tmp_path / "b.h5")

    assert copy.header == header
    assert copy.coordinates.tolist() == coordinates.tolist()
    assert copy.features.shape == features.shape
    assert copy.features.tobytes() == features.tobytes()
    with h5py.File(tmp_path / "b.h5") as file:
        assert file.attrs["tile_count"] == len(features)
