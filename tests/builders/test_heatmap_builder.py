"""Check heatmap processor orchestration, model verification and the heatmap builder."""

from __future__ import annotations

import asyncio
import gc
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Self
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from model_service.builders.application import BuilderApplication
from model_service.builders.contracts import (
    BuilderFailure,
    FailureCode,
    SharedArtifact,
    SharedPathInput,
    SpatialScoresModel,
)
from model_service.builders.heatmap.builder import HeatmapBuilder
from model_service.builders.heatmap.predictor import RayModelPredictor
from model_service.builders.heatmap.processor import (
    HeatmapRequest,
    SlideHeatmapProcessor,
)
from model_service.builders.registry import get_builder
from model_service.builders.score_grid import read_score_grid
from model_service.builders.settings import WorkerSettings
from model_service.providers.shared_storage import SharedStorageAccessAdapter
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


class HeatmapProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_orchestration_score_grid_and_cleanup(self) -> None:
        recorded: dict[str, Any] = {}
        predictor = SimpleNamespace(predict=AsyncMock(return_value=0.75))

        def grid_tiles(**options: object) -> list[tuple[int, int]]:
            recorded["grid_options"] = options
            return [(0, 0)]

        fake_modules = {
            "ratiopath": module("ratiopath"),
            "ratiopath.masks": module("ratiopath.masks"),
            "ratiopath.masks.mask_builders": module(
                "ratiopath.masks.mask_builders",
                MaskBuilder=lambda **options: FakeMaskBuilder(recorded, **options),
            ),
            "ratiopath.openslide": module("ratiopath.openslide", OpenSlide=FakeSlide),
            "ratiopath.tiling": module("ratiopath.tiling", grid_tiles=grid_tiles),
            "model_service.builders.tile_reader": module(
                "model_service.builders.tile_reader",
                TileReader=FakeTileReader,
                open_slide=lambda path, _name: FakeSlide(str(path)),
            ),
        }
        with TemporaryDirectory() as directory, patch.dict(sys.modules, fake_modules):
            output = Path(directory) / "heatmap.npz"
            request = HeatmapRequest(
                slide_path=Path(directory) / "slide.svs",
                mask_path=None,
                output_path=output,
                stride_fraction=0.01,
            )
            processor = SlideHeatmapProcessor(
                predictor, num_threads=1, max_concurrent_tasks=2
            )
            await processor.build(spatial(), request)
            np.testing.assert_array_equal(recorded["batch"], [[[[0.75]]]])
            grid = read_score_grid(output)
            np.testing.assert_array_equal(grid.values, 0.25)
            self.assertEqual(grid.values.shape, (1, 4, 4))
            self.assertTrue(grid.scored.all())
            self.assertEqual(
                (grid.level, grid.level_size, grid.cell_size, grid.mpp),
                (0, (4, 4), 1.0, (0.5, 0.5)),
            )

            predictor.predict.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            await processor.build(spatial(output_tile_size=2), request)
            np.testing.assert_allclose(recorded["batch"], [[[[0.1, 0.2], [0.3, 0.4]]]])

            predictor.predict.return_value = [0.75]
            with self.assertRaises(ValueError):
                await processor.build(spatial(output_tile_size=2), request)
            self.assertTrue(recorded["cleaned"])

            cancelled = asyncio.Event()
            both_started = asyncio.Event()
            calls = 0

            async def failing_prediction(_tile: object) -> object:
                nonlocal calls
                calls += 1
                if calls == 1:
                    await both_started.wait()
                    raise RuntimeError("prediction failed")
                both_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            fake_modules["ratiopath.tiling"].grid_tiles = lambda **_options: [  # type: ignore[attr-defined]
                (0, 0),
                (1, 0),
                (2, 0),
            ]
            predictor.predict.side_effect = failing_prediction
            recorded["check_cleanup"] = lambda: self.assertTrue(cancelled.is_set())
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                await processor.build(spatial(), request)
            self.assertTrue(cancelled.is_set())
            self.assertTrue(recorded["cleaned"])

            recorded.pop("check_cleanup")
            both_started = asyncio.Event()
            calls = 0

            async def simultaneous_failures(_tile: object) -> object:
                nonlocal calls
                calls += 1
                if calls == 2:
                    both_started.set()
                await both_started.wait()
                raise RuntimeError("simultaneous failure")

            predictor.predict.side_effect = simultaneous_failures
            loop = asyncio.get_running_loop()
            previous_handler = loop.get_exception_handler()
            unhandled: list[object] = []
            loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
            try:
                with self.assertRaisesRegex(RuntimeError, "simultaneous failure"):
                    await processor.build(spatial(), request)
                gc.collect()
                await asyncio.sleep(0)
                self.assertEqual(unhandled, [])
            finally:
                loop.set_exception_handler(previous_handler)

        self.assertEqual(recorded["grid_options"]["stride"], (1, 1))
        self.assertEqual(recorded["mask_options"]["stride"], 1)
        np.testing.assert_array_equal(recorded["coords"], [[0, 0]])

    def test_processor_rejects_nonpositive_limits(self) -> None:
        predictor = SimpleNamespace(predict=AsyncMock())
        for limits in ((0, 1), (1, 0)):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                SlideHeatmapProcessor(
                    predictor, num_threads=limits[0], max_concurrent_tasks=limits[1]
                )


def fake_handle(config: dict[str, object], score: object = 0.75) -> SimpleNamespace:
    return SimpleNamespace(
        get_config=SimpleNamespace(remote=AsyncMock(return_value=config)),
        predict=SimpleNamespace(remote=AsyncMock(return_value=score)),
    )


def ray_module(handle: object) -> dict[str, ModuleType]:
    get_handle = Mock(return_value=handle)
    return {"ray": module("ray", serve=SimpleNamespace(get_app_handle=get_handle))}


class RayModelPredictorTests(unittest.IsolatedAsyncioTestCase):
    async def test_verify_accepts_matching_configuration_and_forwards_tiles(
        self,
    ) -> None:
        handle = fake_handle(
            {"tile_size": 4, "output_tile_size": 1, "n_channels": 1, "mpp": 0.5}
        )
        modules = ray_module(handle)
        with patch.dict(sys.modules, modules):
            predictor = RayModelPredictor("classifier")
            modules["ray"].serve.get_app_handle.assert_not_called()
            await predictor.verify(spatial())
            tile = np.zeros((3, 4, 4), dtype=np.uint8)
            self.assertEqual(await predictor.predict(tile), 0.75)
        modules["ray"].serve.get_app_handle.assert_called_once_with("classifier")
        self.assertIs(handle.predict.remote.call_args.args[0], tile)

    async def test_verify_rejects_a_running_model_that_differs(self) -> None:
        valid = {"tile_size": 4, "output_tile_size": 1, "n_channels": 1, "mpp": 0.5}
        for config in (
            valid | {"tile_size": 8},
            valid | {"mpp": 0.25},
            valid | {"n_channels": 2},
            valid | {"tile_size": True},
            valid | {"mpp": float("nan")},
            {"tile_size": 4},
            valid | {"unexpected": 1},
        ):
            with (
                self.subTest(config=config),
                patch.dict(sys.modules, ray_module(fake_handle(config))),
                self.assertRaises(BuilderFailure) as raised,
            ):
                await RayModelPredictor("classifier").verify(spatial())
            self.assertEqual(raised.exception.code, FailureCode.MODEL_CONTRACT_MISMATCH)
            self.assertFalse(raised.exception.retryable)


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
        )
        with patch("ray.serve.get_app_handle", return_value=handle):
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

    async def test_cells_record_their_size_and_whether_they_were_scored(self) -> None:
        self.model = spatial(tile_size=32, output_tile_size=1)
        config = {**self.config, "output_tile_size": 1}

        artifact = await self.run_builder(
            fake_handle(config, np.full((1, 1), 0.5, dtype=np.float32))
        )

        grid = read_score_grid(Path(artifact.path))
        self.assertEqual(grid.cell_size, 16.0)
        self.assertTrue(grid.scored.any())
        np.testing.assert_allclose(grid.values[0][grid.scored], 0.5)

    async def test_missing_mask_processes_the_whole_slide(self) -> None:
        handle = fake_handle(self.config, np.full((32, 32), 0.375, dtype=np.float32))
        artifact = await self.run_builder(handle, mask=False)
        np.testing.assert_allclose(self.scores(artifact), 0.375)
        # 96x64 at tile 32 and stride 16: 5 x 3 grid positions.
        self.assertEqual(handle.predict.remote.await_count, 15)

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

    async def test_contract_mismatch_fails_before_writing(self) -> None:
        handle = fake_handle(self.config | {"tile_size": 64})
        with self.assertRaises(BuilderFailure) as raised:
            await self.run_builder(handle)
        self.assertEqual(raised.exception.code, FailureCode.MODEL_CONTRACT_MISMATCH)
        handle.predict.remote.assert_not_awaited()
        self.assertEqual(list(self.root.glob("*heatmap*")), [])

    async def test_slide_smaller_than_the_model_tile_is_padded(self) -> None:
        self.model = spatial(tile_size=1024, output_tile_size=1024)
        handle = fake_handle(
            self.config | {"tile_size": 1024, "output_tile_size": 1024},
            np.full((1024, 1024), 0.375, dtype=np.float32),
        )
        artifact = await self.run_builder(handle)
        np.testing.assert_allclose(self.scores(artifact), 0.375)


if __name__ == "__main__":
    unittest.main()


class UnreadableInputTests(unittest.TestCase):
    def test_a_file_that_is_not_a_slide_is_an_invalid_input(self) -> None:
        from model_service.builders.contracts import InvalidInvocation
        from model_service.builders.tile_reader import open_slide

        with TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("not a slide")
            with self.assertRaises(InvalidInvocation), open_slide(path, "slide"):
                pass
