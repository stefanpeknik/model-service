"""Where `slide_tiles` reads and where `score_grid` puts each score, through a whole run.

The model is a landmark: it scores a tile with its origin pixel's red value,
or a constant, so a misplaced tile or a wrong overlap mean shows in the grid.
The slides are checkerboards of 16 px squares at 0.5 µm/px; the model reads
32 px tiles, so stride 0.5 puts one tile every 16 px.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest
from shapely import box

from model_service.catalog import load_catalog
from model_service.contracts.failures import InvalidInvocation
from model_service.formats import region_values, regions, score_grid
from model_service.pipeline_runner import run_pipeline
from tests.components.helpers import fake_slide
from tests.support.pipeline_catalog import REGISTRY, model_file, write


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.catalog.pipelines import Pipeline


class Landmark:
    """Scores each tile with its origin's red value, or a constant."""

    def __init__(self, size: int = 1, constant: float | None = None) -> None:
        self.size = size
        self.constant = constant
        self.calls = 0

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        self.calls += 1
        score = tile[0, 0, 0] / 255 if self.constant is None else self.constant
        return np.full((1, self.size, self.size), score, np.float32)


def pipeline(root: Path, *, stride: float = 0.5, size: int = 1) -> Pipeline:
    catalog = root / f"catalog-{stride}-{size}"
    write(
        catalog,
        "models/landmark.yaml",
        model_file(tile_size=32, output=f"kind: scores, size: {size}"),
    )
    write(
        catalog,
        "pipelines/a1.yaml",
        f"""
        source: {{use: slide_tiles, skip_background: false, stride_fraction: {stride}}}
        steps: [{{model: landmark}}]
        sinks: [{{use: score_grid, output: heatmap}}]
        """,
    )
    return load_catalog(catalog, registry=REGISTRY).pipelines["a1"]


def checkerboard(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = ((x // 16 + y // 16) % 2) * 255
    return pixels


# Rows 40-59 and columns 50-89 of a 112 x 80 slide, in level-0 pixels.
REGION = box(50, 40, 90, 60)


def heatmap(
    root: Path,
    width: int,
    height: int,
    *,
    model: Landmark | None = None,
    stride: float = 0.5,
    roi: str = "none",
) -> score_grid.ScoreGrid:
    """Run the pipeline on a checkerboard slide and read the grid it wrote."""
    model = model or Landmark()
    directory = root / f"{width}-{height}-{stride}-{model.size}-{roi}"
    directory.mkdir()
    pixels = checkerboard(width, height)
    inputs = {"slide": fake_slide(directory / "slide.tiff", pixels)}
    if roi != "none":
        shapes = {
            "empty": [],
            "full": [box(0, 0, width, height)],
            "region": [REGION],
        }[roi]
        inputs["region_of_interest"] = directory / "regions.geojson"
        regions.write(
            inputs["region_of_interest"],
            tuple(regions.Region(f"r{i}", shape) for i, shape in enumerate(shapes)),
        )
    output = directory / "heatmap.npz"
    asyncio.run(
        run_pipeline(
            pipeline(root, stride=stride, size=model.size),
            inputs,
            {"heatmap": output},
            {"landmark": model},
        )
    )
    grid = score_grid.read(output)
    assert (grid.level, grid.level_size) == (0, (width, height))
    assert grid.mpp == (0.5, 0.5)
    assert grid.values.dtype == np.float32
    return grid


def render(grid: score_grid.ScoreGrid) -> np.ndarray:
    """The level's pixels, each with the score of the cell it falls in."""
    width, height = grid.level_size
    rows = (np.arange(height) / grid.cell_size).astype(int)
    columns = (np.arange(width) / grid.cell_size).astype(int)
    return grid.values[0][np.ix_(rows, columns)]


@pytest.mark.parametrize(
    ("width", "height", "stride", "size"),
    [
        (112, 80, 0.5, 1),
        (128, 96, 1.0, 1),
        (112, 82, 0.3, 1),
        (112, 80, 0.5, 2),
        (112, 80, 0.5, 3),
        (112, 80, 0.5, 64),
    ],
)
def test_padding_does_not_move_tile_placement(
    tmp_path: Path, width: int, height: int, stride: float, size: int
) -> None:
    aligned = heatmap(tmp_path, width, height, stride=stride, model=Landmark(size))
    unaligned = heatmap(tmp_path, 107, 75, stride=stride, model=Landmark(size))

    np.testing.assert_array_equal(render(unaligned), render(aligned)[:75, :107])


def test_cells_record_whether_any_tile_scored_them(tmp_path: Path) -> None:
    scored = heatmap(tmp_path, 112, 80, model=Landmark(constant=0.5))
    unscored = heatmap(tmp_path, 112, 80, model=Landmark(constant=0.5), roi="empty")

    # A tile every 16 px yields one score per 16 px; an empty region list scores none.
    assert scored.cell_size == 16.0
    assert scored.scored.any()
    np.testing.assert_allclose(scored.values[0][scored.scored], 0.5)
    assert not unscored.scored.any()


def test_scores_land_on_their_tiles_and_overlaps_are_averaged(tmp_path: Path) -> None:
    pixels = render(heatmap(tmp_path, 112, 80))

    assert pixels[0, 0] == 0
    assert pixels[0, -1] == 1
    assert pixels[-1, 0] == 1
    assert pixels[-1, -1] == 0
    # Inside, four tiles overlap a cell: two black and two red origins.
    assert pixels[39, 55] == 0.5


def test_constant_scores_remain_constant_through_overlap_and_padding(
    tmp_path: Path,
) -> None:
    model = Landmark(constant=0.375)

    pixels = render(heatmap(tmp_path, 107, 75, model=model))

    assert model.calls == 24
    np.testing.assert_array_equal(pixels, np.full((75, 107), 0.375))


def test_a_region_over_the_whole_slide_scores_like_none(tmp_path: Path) -> None:
    masked, unmasked = Landmark(), Landmark()

    with_mask = heatmap(tmp_path, 107, 75, model=masked, roi="full")
    without_mask = heatmap(tmp_path, 107, 75, model=unmasked)

    assert (masked.calls, unmasked.calls) == (24, 24)
    np.testing.assert_array_equal(without_mask.values, with_mask.values)


def test_a_region_of_interest_limits_which_tiles_are_scored(tmp_path: Path) -> None:
    whole_model, region_model = Landmark(), Landmark()

    whole = heatmap(tmp_path, 112, 80, model=whole_model)
    region = heatmap(tmp_path, 112, 80, model=region_model, roi="region")

    assert region.values.shape == whole.values.shape
    assert region_model.calls < whole_model.calls
    # The box's cells, rows 2-3 and columns 3-5, score as in a whole-slide run.
    np.testing.assert_array_equal(region.values[:, 2:4, 3:6], whole.values[:, 2:4, 3:6])
    assert region.scored[2:4, 3:6].all()
    assert not region.scored[:, 0].any()


def test_an_empty_region_list_scores_nothing(tmp_path: Path) -> None:
    model = Landmark()

    grid = heatmap(tmp_path, 107, 75, model=model, roi="empty")

    assert model.calls == 0
    assert not grid.values.any()
    assert not grid.scored.any()


def test_tiles_are_read_at_the_level_closest_to_the_models_resolution(
    tmp_path: Path,
) -> None:
    slide = fake_slide(
        tmp_path / "slide.tiff", checkerboard(256, 192), mpp=0.25, pyramid=True
    )
    output = tmp_path / "heatmap.npz"

    asyncio.run(
        run_pipeline(
            pipeline(tmp_path),
            {"slide": slide},
            {"heatmap": output},
            {"landmark": Landmark(constant=0.5)},
        )
    )

    grid = score_grid.read(output)
    assert (grid.level, grid.level_size, grid.mpp) == (1, (128, 96), (0.5, 0.5))


def test_a_file_that_is_not_a_slide_is_an_invalid_input(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("not a slide")

    with pytest.raises(InvalidInvocation):
        asyncio.run(
            run_pipeline(
                pipeline(tmp_path),
                {"slide": notes},
                {"heatmap": tmp_path / "heatmap.npz"},
                {"landmark": Landmark()},
            )
        )


def test_tiles_of_every_region_are_scored_and_none_between(tmp_path: Path) -> None:
    directory = tmp_path / "two"
    directory.mkdir()
    slide = fake_slide(directory / "slide.tiff", checkerboard(112, 80))
    roi = directory / "regions.geojson"
    regions.write(
        roi,
        (
            regions.Region("left", box(0, 0, 16, 16)),
            regions.Region("right", box(96, 64, 112, 80)),
        ),
    )
    output = directory / "heatmap.npz"
    model = Landmark(constant=0.5)

    asyncio.run(
        run_pipeline(
            pipeline(tmp_path),
            {"slide": slide, "region_of_interest": roi},
            {"heatmap": output},
            {"landmark": model},
        )
    )

    grid = score_grid.read(output)
    # 32 px tiles every 16 px: one tile covers each corner box.
    assert model.calls == 2
    assert grid.scored[0, 0] and grid.scored[4, 6]
    assert not grid.scored[2, 3]


def test_regions_are_in_level_zero_pixels_when_a_coarser_level_is_read(
    tmp_path: Path,
) -> None:
    slide = fake_slide(
        tmp_path / "slide.tiff", checkerboard(256, 192), mpp=0.25, pyramid=True
    )
    roi = tmp_path / "regions.geojson"
    # The left half of level 0 is the left half of level 1 too: the tile at
    # x 48 overlaps it and scores cells 3 and 4; the one at 64 only touches it.
    regions.write(roi, (regions.Region("half", box(0, 0, 128, 192)),))
    output = tmp_path / "heatmap.npz"

    asyncio.run(
        run_pipeline(
            pipeline(tmp_path),
            {"slide": slide, "region_of_interest": roi},
            {"heatmap": output},
            {"landmark": Landmark(constant=0.5)},
        )
    )

    grid = score_grid.read(output)
    assert grid.level == 1
    assert grid.scored[:, :5].all()
    assert not grid.scored[:, 5:].any()


def test_a_region_file_that_is_not_geojson_is_an_invalid_input(tmp_path: Path) -> None:
    slide = fake_slide(tmp_path / "slide.tiff", checkerboard(112, 80))
    roi = tmp_path / "regions.geojson"
    roi.write_text("not json")

    with pytest.raises(InvalidInvocation):
        asyncio.run(
            run_pipeline(
                pipeline(tmp_path),
                {"slide": slide, "region_of_interest": roi},
                {"heatmap": tmp_path / "heatmap.npz"},
                {"landmark": Landmark()},
            )
        )


def test_the_regions_reach_the_sinks_with_their_ids(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog-regions"
    write(
        catalog,
        "models/landmark.yaml",
        model_file(tile_size=32, output="kind: scores, size: 2"),
    )
    write(
        catalog,
        "pipelines/a1.yaml",
        """
        source: {use: slide_tiles, skip_background: false}
        steps: [{model: landmark}]
        sinks: [{use: region_fraction, output: purity, threshold: 0.5}]
        """,
    )
    slide = fake_slide(tmp_path / "slide.tiff", checkerboard(112, 80))
    roi = tmp_path / "regions.geojson"
    # Two tiles overlap the box, each with one 16 px score pixel in it: the
    # tile at x 0 scores 0 (its origin is black), the one at x 16 scores 1.
    regions.write(
        roi,
        (
            regions.Region("box", box(16, 0, 32, 16)),
            regions.Region("outside", box(200, 200, 210, 210)),
        ),
    )
    output = tmp_path / "purity.json"

    asyncio.run(
        run_pipeline(
            load_catalog(catalog, registry=REGISTRY).pipelines["a1"],
            {"slide": slide, "region_of_interest": roi},
            {"purity": output},
            {"landmark": Landmark(size=2)},
        )
    )

    assert [(r.id, r.value) for r in region_values.read(output)] == [
        ("box", 0.5),
        ("outside", None),
    ]
