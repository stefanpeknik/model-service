"""Check heatmap processor orchestration, model verification and the heatmap builder."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Self
from unittest.mock import AsyncMock

import numpy as np

from model_service.builders.application import BuilderApplication
from model_service.builders.contracts import (
    SharedArtifact,
    SharedPathInput,
    SpatialScoresModel,
)
from model_service.builders.heatmap.builder import HeatmapBuilder
from model_service.builders.heatmap.score_grid import read_score_grid
from model_service.builders.registry import get_builder
from model_service.builders.settings import WorkerSettings
from model_service.providers.shared_storage import SharedStorageAccessAdapter
from tests.support.builder_contract import BuilderContractTests
from tests.support.builder_fixtures import invocation, write_slide


if TYPE_CHECKING:
    from collections.abc import Iterator


def spatial(**overrides: Any) -> SpatialScoresModel:
    values: dict[str, Any] = {
        "id": "classifier",
        "tile_size": 4,
        "output_tile_size": 1,
        "n_channels": 1,
        "mpp": 0.5,
    }
    return SpatialScoresModel(**(values | overrides))


class FakeSlide:
    level_dimensions = ((4, 4),)

    def __init__(self, _path: str) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def slide_resolution(self, _level: int) -> tuple[float, float]:
        return 0.5, 0.5


class FakeTileReader:
    level = 0
    tissue = None

    @classmethod
    @contextmanager
    def open(
        cls, _slide: object, _mask: object, _mpp: float, *, skip_background: bool
    ) -> Iterator[Self]:
        yield cls()

    def read(self, _x: int, _y: int, tile_size: int) -> np.ndarray:
        return np.zeros((3, tile_size, tile_size), dtype=np.uint8)


class FakeMaskBuilder:
    def __init__(self, recorded: dict[str, Any], **options: object) -> None:
        self.recorded = recorded
        recorded["mask_options"] = options
        self.span = np.asarray(options["source_extents"])

    def update_batch(self, *, batch: np.ndarray, coords: np.ndarray) -> None:
        self.recorded["batch"] = batch
        self.recorded["coords"] = coords

    def finalize(self) -> dict[str, np.ndarray]:
        return {
            "mask": np.full((1, 4, 4), 0.25, dtype=np.float32),
            "overlap_counter": np.ones((1, 4, 4)),
        }

    def cleanup(self) -> None:
        if "check_cleanup" in self.recorded:
            self.recorded["check_cleanup"]()
        self.recorded["cleaned"] = True


def module(name: str, **members: object) -> ModuleType:
    result = ModuleType(name)
    for key, value in members.items():
        setattr(result, key, value)
    return result


def fake_handle(config: dict[str, object], score: object = 0.75) -> SimpleNamespace:
    return SimpleNamespace(
        get_config=SimpleNamespace(remote=AsyncMock(return_value=config)),
        predict=SimpleNamespace(remote=AsyncMock(return_value=score)),
    )


class HeatmapContractTests(BuilderContractTests, unittest.IsolatedAsyncioTestCase):
    builder_id = "heatmap"
    model = spatial(tile_size=32, output_tile_size=4)


class HeatmapBuilderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        # H&E-like pink everywhere, so the background filter sees tissue.
        pixels = np.empty((64, 96, 3), dtype=np.uint8)
        pixels[...] = (200, 120, 160)
        write_slide(self.root / "slide.tiff", pixels)
        write_slide(self.root / "mask.tiff", np.full_like(pixels, 255))
        self.model = spatial(tile_size=32, output_tile_size=32)
        self.config = {
            "tile_size": 32,
            "output_tile_size": 32,
            "n_channels": 1,
            "mpp": 0.5,
        }

    async def run_builder(
        self,
        handle: object,
        *,
        configuration: dict[str, object] | None = None,
        mask: bool = True,
    ) -> SharedArtifact:
        inputs = {"slide": SharedPathInput("slide.tiff")}
        if mask:
            inputs["region_of_interest"] = SharedPathInput("mask.tiff")
        application = BuilderApplication(
            get_builder("heatmap"),
            HeatmapBuilder(WorkerSettings(num_threads=1, max_concurrent_tasks=2)),
            {"shared_storage": SharedStorageAccessAdapter([self.root])},
            model_handles=lambda _model_id: handle,  # type: ignore[arg-type, return-value]
        )
        result = await application.run(
            invocation(
                "shared_storage",
                {"root": str(self.root)},
                inputs=inputs,
                resolved_configuration=configuration,
            ),
            self.model,
        )
        artifact = result.outputs["heatmap"]
        assert isinstance(artifact, SharedArtifact)
        return artifact

    def scores(self, artifact: SharedArtifact) -> np.ndarray:
        """The slide's 96x64 pixels' scores; these models score every pixel."""
        grid = read_score_grid(Path(artifact.path))
        self.assertEqual(grid.cell_size, 1.0)
        self.assertEqual(grid.level_size, (96, 64))
        return grid.values[0, :64, :96]

    async def test_the_score_grid_is_written_at_the_default_path(self) -> None:
        artifact = await self.run_builder(
            fake_handle(self.config, np.full((32, 32), 0.375, dtype=np.float32))
        )
        self.assertEqual(artifact.path, str(self.root.resolve() / "heatmap.npz"))
        self.assertEqual(artifact.media_type, "application/x-npz")
        np.testing.assert_allclose(self.scores(artifact), 0.375)
        self.assertEqual(list(self.root.glob(".*")), [])

    async def test_background_tiles_are_skipped_unless_disabled(self) -> None:
        pixels = np.full((64, 96, 3), 245, dtype=np.uint8)  # white glass
        pixels[:, :48] = (200, 120, 160)  # tissue on the left half
        write_slide(self.root / "slide.tiff", pixels)
        cases = {
            # Columns x = 0, 16, 32 touch the left half, in 3 rows.
            "default": ({}, 9),
            "disabled": ({"skip_background": False}, 15),
        }
        for case, (configuration, calls) in cases.items():
            with self.subTest(case):
                handle = fake_handle(
                    self.config, np.full((32, 32), 0.375, dtype=np.float32)
                )
                artifact = await self.run_builder(
                    handle, configuration=configuration, mask=False
                )
                self.assertEqual(handle.predict.remote.await_count, calls)
                scores = self.scores(artifact)
                np.testing.assert_allclose(scores[:, :48], 0.375)
                if case == "default":
                    np.testing.assert_array_equal(scores[:, 64:], 0)

    async def test_slide_smaller_than_the_model_tile_is_padded(self) -> None:
        self.model = spatial(tile_size=1024, output_tile_size=1024)
        handle = fake_handle(
            self.config | {"tile_size": 1024, "output_tile_size": 1024},
            np.full((1024, 1024), 0.375, dtype=np.float32),
        )
        artifact = await self.run_builder(handle)
        np.testing.assert_allclose(self.scores(artifact), 0.375)


class UnreadableInputTests(unittest.TestCase):
    def test_a_file_that_is_not_a_slide_is_an_invalid_input(self) -> None:
        from model_service.builders.contracts import InvalidInvocation
        from model_service.builders.slides.tile_reader import open_slide

        with TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("not a slide")
            with self.assertRaises(InvalidInvocation), open_slide(path, "slide"):
                pass


if __name__ == "__main__":
    unittest.main()
