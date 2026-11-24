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

from model_service.catalog import load_catalog
from model_service.contracts.failures import InvalidInvocation
from model_service.formats import score_grid
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


# Rows 40-59 and columns 50-89 of a 112 x 80 slide.
REGION = (slice(40, 60), slice(50, 90))


def heatmap(
    root: Path,
    width: int,
    height: int,
    *,
    model: Landmark | None = None,
    stride: float = 0.5,
    mask: str = "none",
    mask_pixels: np.ndarray | None = None,
) -> score_grid.ScoreGrid:
    """Run the pipeline on a checkerboard slide and read the grid it wrote."""
    model = model or Landmark()
    directory = root / f"{width}-{height}-{stride}-{model.size}-{mask}"
    directory.mkdir()
    pixels = checkerboard(width, height)
    inputs = {"slide": fake_slide(directory / "slide.tiff", pixels)}
    if mask != "none" or mask_pixels is not None:
        selected = np.full_like(pixels, 0 if mask == "empty" else 255)
        if mask == "region":
            selected[:] = 0
            selected[REGION] = 255
        inputs["region_of_interest"] = fake_slide(
            directory / "mask.tiff", selected if mask_pixels is None else mask_pixels
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
    unscored = heatmap(tmp_path, 112, 80, model=Landmark(constant=0.5), mask="empty")

    # A tile every 16 px yields one score per 16 px; the empty mask scores none.
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


def test_a_full_mask_scores_like_no_mask(tmp_path: Path) -> None:
    masked, unmasked = Landmark(), Landmark()

    with_mask = heatmap(tmp_path, 107, 75, model=masked, mask="full")
    without_mask = heatmap(tmp_path, 107, 75, model=unmasked)

    assert (masked.calls, unmasked.calls) == (24, 24)
    np.testing.assert_array_equal(without_mask.values, with_mask.values)


def test_a_region_of_interest_limits_which_tiles_are_scored(tmp_path: Path) -> None:
    whole_model, region_model = Landmark(), Landmark()

    whole = heatmap(tmp_path, 112, 80, model=whole_model)
    region = heatmap(tmp_path, 112, 80, model=region_model, mask="region")

    assert region.values.shape == whole.values.shape
    assert region_model.calls < whole_model.calls
    # The box's cells, rows 2-3 and columns 3-5, score as in a whole-slide run.
    np.testing.assert_array_equal(region.values[:, 2:4, 3:6], whole.values[:, 2:4, 3:6])
    assert region.scored[2:4, 3:6].all()
    assert not region.scored[:, 0].any()


def test_a_coarser_mask_selects_tiles_by_its_truncated_window(tmp_path: Path) -> None:
    # A mask at a third of the slide's size: a 32 px tile at x covers mask
    # columns int(x / 3) to int(x / 3) + int(32 / 3), as in the original
    # model-service tiling. Mask pixel (15, 15) is in the window of the tile
    # at (32, 32) only; the tile at 16 covers columns 5 to 14.
    mask_pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    mask_pixels[15, 15] = 255
    model = Landmark()

    heatmap(tmp_path, 96, 96, model=model, mask_pixels=mask_pixels)

    assert model.calls == 1


def test_an_empty_mask_scores_nothing(tmp_path: Path) -> None:
    model = Landmark()

    grid = heatmap(tmp_path, 107, 75, model=model, mask="empty")

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
