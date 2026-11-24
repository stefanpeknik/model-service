"""The `value` data format: one JSON scalar per slide, for EMPAIA primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from model_service.formats import score_grid, value
from model_service.formats.known import format_of, get_format
from model_service.formats.media_type import UnsupportedFormat


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("scalar", [True, 3, 0.25, "benign"])
def test_a_scalar_reads_back_as_written(tmp_path: Path, scalar: object) -> None:
    path = tmp_path / "score.json"

    value.write(path, scalar)  # type: ignore[arg-type]

    assert value.read(path) == scalar


def test_a_file_that_holds_no_scalar_is_refused(tmp_path: Path) -> None:
    listed = tmp_path / "listed.json"
    listed.write_text('{"value": [1, 2]}')

    with pytest.raises(ValueError, match="not a bool, number or string"):
        value.read(listed)


def test_an_unknown_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "score.json"
    value.write(path, 1)

    with pytest.raises(UnsupportedFormat):
        value.read(path, "application/vnd.rationai.value+json; version=2")


def test_a_stored_media_type_names_its_format() -> None:
    assert format_of(value.MEDIA_TYPE) is get_format("value")
    assert format_of(score_grid.MEDIA_TYPE) is get_format("score_grid")
    assert format_of("application/x-npz") is None
    assert format_of("image/tiff") is None
