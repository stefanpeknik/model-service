"""Check the heatmap builder against real slides and a fake model."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import numpy as np

from model_service.builders.base import ReplicaSettings
from model_service.builders.implementations.heatmap.builder import HeatmapBuilder
from model_service.builders.registry import get_builder
from model_service.builders.replica.runner import BuilderApplication
from model_service.contracts.invocation import SharedPathInput
from model_service.contracts.model_declarations import SpatialScoresModel
from model_service.contracts.result import SharedArtifact
from model_service.contracts.score_grid import read_score_grid
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from tests.support.builder_contract import BuilderContractTests
from tests.support.builder_fixtures import invocation, write_slide


def spatial(**overrides: Any) -> SpatialScoresModel:
    values: dict[str, Any] = {
        "id": "classifier",
        "tile_size": 4,
        "output_tile_size": 1,
        "n_channels": 1,
        "mpp": 0.5,
    }
    return SpatialScoresModel(**(values | overrides))


def fake_handle(config: dict[str, object], score: object = 0.75) -> SimpleNamespace:
    return SimpleNamespace(
        get_config=SimpleNamespace(remote=AsyncMock(return_value=config)),
        predict=SimpleNamespace(remote=AsyncMock(return_value=score)),
    )


class HeatmapContractTests(BuilderContractTests, unittest.IsolatedAsyncioTestCase):
    builder_id = "heatmap"
    # Tissue detection needs a realistic slide; see slides/test_tissue.py.
    configuration: ClassVar[dict[str, Any]] = {"skip_background": False}
    model = spatial(tile_size=32, output_tile_size=4)


class HeatmapBuilderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
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
            HeatmapBuilder(ReplicaSettings(num_threads=1, max_concurrent_tasks=2)),
            {"shared_storage": SharedStorageAccessFactory([self.root])},
            model_handles=lambda _model_id: handle,  # type: ignore[arg-type, return-value]
        )
        result = await application.run(
            invocation(
                "shared_storage",
                {"root": str(self.root)},
                inputs=inputs,
                # Tissue detection needs a realistic slide; see slides/test_tissue.py.
                configuration={"skip_background": False} | (configuration or {}),
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
        from model_service.builders.slides.tile_reader import open_slide
        from model_service.contracts.failures import InvalidInvocation

        with TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("not a slide")
            with self.assertRaises(InvalidInvocation), open_slide(path, "slide"):
                pass


if __name__ == "__main__":
    unittest.main()
