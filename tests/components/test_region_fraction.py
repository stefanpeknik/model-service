"""`region_fraction`: the share of each region's scored pixels at or above a threshold."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely import box

from components.sinks.region_fraction import RegionFraction
from model_service.formats import region_values
from model_service.formats.regions import Region
from tests.components.helpers import fake_frame, run_component


if TYPE_CHECKING:
    from pathlib import Path


def half_positive_tile(x: int = 0, y: int = 0, value: float = 1.0) -> dict:
    """One 64 px tile whose 4 x 4 score map is `value` on the left half, 0 on the right."""
    scores = np.zeros((1, 1, 4, 4), np.float32)
    scores[0, 0, :, :2] = value
    return {
        "tile_x": np.array([x], np.int64),
        "tile_y": np.array([y], np.int64),
        "scores": scores,
    }


def fractions(
    tmp_path: Path,
    regions: tuple[Region, ...] | None,
    batch: dict | None = None,
    **options: object,
) -> list[tuple[str, float | None]]:
    path = run_component(
        RegionFraction,
        [batch or half_positive_tile()],
        options={"output": "purity", **options},
        work_dir=tmp_path,
        frame=fake_frame(regions=regions),
    )
    return [(r.id, r.value) for r in region_values.read(path)]


def test_a_region_over_the_whole_tile_gets_the_tiles_share(tmp_path: Path) -> None:
    assert fractions(tmp_path, (Region("a", box(0, 0, 64, 64)),)) == [("a", 0.5)]


def test_each_region_counts_only_the_pixels_inside_it(tmp_path: Path) -> None:
    found = fractions(
        tmp_path,
        (Region("left", box(0, 0, 32, 64)), Region("right", box(32, 0, 64, 64))),
    )

    assert found == [("left", 1.0), ("right", 0.0)]


def test_a_region_no_tile_covers_has_no_value(tmp_path: Path) -> None:
    assert fractions(tmp_path, (Region("far", box(200, 200, 250, 250)),)) == [
        ("far", None)
    ]


def test_regions_are_in_level_zero_pixels(tmp_path: Path) -> None:
    # At downsample 2, level-0 x 0..64 is level x 0..32: the tile's left half.
    path = run_component(
        RegionFraction,
        [half_positive_tile()],
        options={"output": "purity"},
        work_dir=tmp_path,
        frame=fake_frame(downsample=2.0, regions=(Region("a", box(0, 0, 64, 128)),)),
    )

    assert [(r.id, r.value) for r in region_values.read(path)] == [("a", 1.0)]


def test_without_regions_the_whole_slide_is_one_region(tmp_path: Path) -> None:
    assert fractions(tmp_path, None) == [("slide", 0.5)]


def test_scores_below_the_threshold_do_not_count(tmp_path: Path) -> None:
    found = fractions(
        tmp_path,
        (Region("a", box(0, 0, 64, 64)),),
        half_positive_tile(value=0.6),
        threshold=0.7,
    )

    assert found == [("a", 0.0)]
