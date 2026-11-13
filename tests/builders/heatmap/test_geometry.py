"""Check tile placement, overlap means, padding and masks with native image libraries."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from model_service.builders.heatmap.processor import build_heatmap
from model_service.contracts.model_declarations import SpatialScoresModel
from model_service.contracts.score_grid import ScoreGrid, read_score_grid
from tests.support.builder_fixtures import write_slide


class LandmarkPredictor:
    """Return a constant score, or the tile's origin pixel, to expose placement errors."""

    def __init__(
        self, output_tile_size: int = 1, constant: float | None = None
    ) -> None:
        self.output_tile_size = output_tile_size
        self.constant = constant
        self.calls = 0
        self.declaration = SpatialScoresModel(
            id="landmark",
            tile_size=32,
            output_tile_size=output_tile_size,
            n_channels=1,
            mpp=0.5,
        )

    async def predict(self, tile: object) -> np.ndarray:
        self.calls += 1
        score = self.constant
        if score is None:
            score = float(np.asarray(tile)[0, 0, 0]) / 255
        return np.full((self.output_tile_size, self.output_tile_size), score)


def render(grid: ScoreGrid) -> np.ndarray:
    """The level's pixels, each with the score of the cell it falls in."""
    width, height = grid.level_size
    rows = (np.arange(height) / grid.cell_size).astype(int)
    columns = (np.arange(width) / grid.cell_size).astype(int)
    return grid.values[0][np.ix_(rows, columns)]


# Rows 40-59 and columns 50-89 of a 112x80 slide.
REGION = (slice(40, 60), slice(50, 90))


class HeatmapGeometryTests(unittest.IsolatedAsyncioTestCase):
    async def _build(
        self,
        root: Path,
        width: int,
        height: int,
        *,
        stride_fraction: float = 0.5,
        predictor: LandmarkPredictor | None = None,
        mask: str = "full",
    ) -> ScoreGrid:
        predictor = predictor or LandmarkPredictor()
        directory = root / f"{width}-{height}-{mask}"
        directory.mkdir()
        y, x = np.indices((height, width))
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = ((x // 16 + y // 16) % 2) * 255
        write_slide(directory / "slide.tiff", pixels)
        mask_path = None
        if mask != "none":
            mask_path = directory / "mask.tiff"
            selected = np.full_like(pixels, 0 if mask == "empty" else 255)
            if mask == "region":
                selected[:] = 0
                selected[REGION] = 255
            write_slide(mask_path, selected)
        output = directory / "heatmap.npz"
        await build_heatmap(
            directory / "slide.tiff",
            mask_path,
            output,
            client=predictor,  # type: ignore[arg-type]
            stride_fraction=stride_fraction,
            skip_background=False,
            num_threads=1,
            max_concurrent_tasks=2,
        )
        grid = read_score_grid(output)
        self.assertEqual((grid.level, grid.level_size), (0, (width, height)))
        self.assertEqual(grid.mpp, (0.5, 0.5))
        self.assertEqual(grid.values.dtype, np.float32)
        return grid

    async def test_padding_does_not_move_tile_placement(self) -> None:
        for width, height, stride, output_tile_size in (
            (112, 80, 0.5, 1),
            (128, 96, 1.0, 1),
            (112, 82, 0.3, 1),
            (112, 80, 0.5, 2),
            (112, 80, 0.5, 3),
            (112, 80, 0.5, 64),
        ):
            with (
                self.subTest(stride=stride, output_tile_size=output_tile_size),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                aligned = await self._build(
                    root,
                    width,
                    height,
                    stride_fraction=stride,
                    predictor=LandmarkPredictor(output_tile_size),
                )
                unaligned = await self._build(
                    root,
                    107,
                    75,
                    stride_fraction=stride,
                    predictor=LandmarkPredictor(output_tile_size),
                )
                np.testing.assert_array_equal(
                    render(unaligned), render(aligned)[:75, :107]
                )

    async def test_cells_record_whether_any_tile_scored_them(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            constant = LandmarkPredictor(constant=0.5)
            scored = await self._build(root, 112, 80, predictor=constant)
            unscored = await self._build(
                root, 112, 80, predictor=constant, mask="empty"
            )

        # A tile every 16 px yields one score per 16 px; the empty mask scores none.
        self.assertEqual(scored.cell_size, 16.0)
        self.assertTrue(scored.scored.any())
        np.testing.assert_allclose(scored.values[0][scored.scored], 0.5)
        self.assertFalse(unscored.scored.any())

    async def test_scores_land_on_their_tiles_and_overlaps_are_averaged(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            pixels = render(await self._build(Path(directory), 112, 80))

        self.assertEqual(pixels[0, 0], 0)
        self.assertEqual(pixels[0, -1], 1)
        self.assertEqual(pixels[-1, 0], 1)
        self.assertEqual(pixels[-1, -1], 0)
        # Inside, four tiles overlap a cell: two black and two red origins.
        self.assertEqual(pixels[39, 55], 0.5)

    async def test_constant_scores_remain_constant_through_overlap_and_padding(
        self,
    ) -> None:
        predictor = LandmarkPredictor(constant=0.375)
        with TemporaryDirectory() as directory:
            pixels = render(
                await self._build(Path(directory), 107, 75, predictor=predictor)
            )

        self.assertEqual(predictor.calls, 24)
        np.testing.assert_array_equal(pixels, np.full((75, 107), 0.375))

    async def test_without_a_mask_every_grid_tile_is_predicted(self) -> None:
        masked = LandmarkPredictor()
        unmasked = LandmarkPredictor()
        with TemporaryDirectory() as directory:
            with_mask = await self._build(Path(directory), 107, 75, predictor=masked)
            without_mask = await self._build(
                Path(directory), 107, 75, predictor=unmasked, mask="none"
            )
        self.assertEqual(unmasked.calls, 24)
        self.assertEqual(masked.calls, 24)
        np.testing.assert_array_equal(without_mask.values, with_mask.values)

    async def test_a_region_of_interest_limits_which_tiles_are_scored(self) -> None:
        whole_predictor, region_predictor = LandmarkPredictor(), LandmarkPredictor()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            whole = await self._build(root, 112, 80, predictor=whole_predictor)
            region = await self._build(
                root, 112, 80, predictor=region_predictor, mask="region"
            )

        self.assertEqual(region.values.shape, whole.values.shape)
        self.assertLess(region_predictor.calls, whole_predictor.calls)
        # The box's cells, rows 2-3 and columns 3-5, score as in a whole-slide run.
        np.testing.assert_array_equal(
            region.values[:, 2:4, 3:6], whole.values[:, 2:4, 3:6]
        )
        self.assertTrue(region.scored[2:4, 3:6].all())
        self.assertFalse(region.scored[:, 0].any())

    async def test_empty_mask_skips_predictions_and_scores_nothing(self) -> None:
        predictor = LandmarkPredictor()
        with TemporaryDirectory() as directory:
            grid = await self._build(
                Path(directory), 107, 75, predictor=predictor, mask="empty"
            )

        self.assertEqual(predictor.calls, 0)
        self.assertFalse(grid.values.any())
        self.assertFalse(grid.scored.any())


if __name__ == "__main__":
    unittest.main()
