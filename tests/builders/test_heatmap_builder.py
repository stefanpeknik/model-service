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
from model_service.builders.registry import ScoreEncoding, get_builder
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


class FakeVipsImage:
    def __init__(self, recorded: dict[str, Any]) -> None:
        self.recorded = recorded

    def __mul__(self, _value: int) -> FakeVipsImage:
        return self

    def resize(self, _scale: float, *, vscale: float) -> FakeVipsImage:
        return self

    def crop(self, _x: int, _y: int, _width: int, _height: int) -> FakeVipsImage:
        return self

    def cast(self, band_format: object) -> FakeVipsImage:
        self.recorded["format"] = band_format
        return self

    def set_type(self, _gtype: object, name: str, value: str) -> None:
        self.recorded[name] = value

    def tiffsave(self, path: str, **options: object) -> None:
        self.recorded["tiff_options"] = options
        Path(path).write_bytes(b"tiff")


class FakeMaskBuilder:
    def __init__(self, recorded: dict[str, Any], **options: object) -> None:
        self.recorded = recorded
        recorded["mask_options"] = options
        self.span = np.asarray(options["source_extents"])

    def update_batch(self, *, batch: np.ndarray, coords: np.ndarray) -> None:
        self.recorded["batch"] = batch
        self.recorded["coords"] = coords

    def finalize(self) -> dict[str, np.ndarray]:
        return {"mask": np.zeros((1, 4, 4), dtype=np.float32)}

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
    async def test_orchestration_encoding_and_cleanup(self) -> None:
        recorded: dict[str, Any] = {}
        predictor = SimpleNamespace(predict=AsyncMock(return_value=0.75))

        def grid_tiles(**options: object) -> list[tuple[int, int]]:
            recorded["grid_options"] = options
            return [(0, 0)]

        fake_modules = {
            "pyvips": module(
                "pyvips",
                Image=SimpleNamespace(
                    new_from_array=lambda _value: FakeVipsImage(recorded)
                ),
                BandFormat=SimpleNamespace(UCHAR="uchar", FLOAT="float"),
                GValue=SimpleNamespace(gstr_type="gchararray"),
                enums=SimpleNamespace(
                    ForeignTiffCompression=SimpleNamespace(DEFLATE="deflate")
                ),
            ),
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
            output = Path(directory) / "heatmap.tiff"
            request = HeatmapRequest(
                slide_path=Path(directory) / "slide.svs",
                mask_path=None,
                output_path=output,
                stride_fraction=0.01,
                output_tile_height=128,
                output_tile_width=256,
            )
            processor = SlideHeatmapProcessor(
                predictor, num_threads=1, max_concurrent_tasks=2
            )
            await processor.build(spatial(), request)
            self.assertEqual(output.read_bytes(), b"tiff")
            self.assertEqual(recorded["format"], "uchar")
            np.testing.assert_array_equal(recorded["batch"], [[[[0.75]]]])

            await processor.build(
                spatial(),
                HeatmapRequest(
                    slide_path=request.slide_path,
                    mask_path=None,
                    output_path=output,
                    stride_fraction=0.01,
                    output_tile_height=128,
                    output_tile_width=256,
                    encoding=ScoreEncoding.FLOAT32,
                ),
            )
            self.assertEqual(recorded["format"], "float")

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
        self.assertEqual(recorded["tiff_options"]["tile_width"], 256)
        self.assertEqual(recorded["tiff_options"]["tile_height"], 128)
        self.assertEqual(recorded["tiff_options"]["xres"], 2000)

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

    def pixels(self, artifact: SharedArtifact) -> tuple[str, np.ndarray]:
        import pyvips

        image = pyvips.Image.new_from_file(artifact.path)
        self.assertEqual((image.width, image.height, image.bands), (96, 64, 1))
        dtype = np.float32 if image.format == "float" else np.uint8
        return image.format, np.frombuffer(image.write_to_memory(), dtype=dtype)

    async def test_default_encoding_is_uint8_at_the_default_path(self) -> None:
        artifact = await self.run_builder(
            fake_handle(self.config, np.full((32, 32), 0.5, dtype=np.float32))
        )
        self.assertEqual(artifact.path, str(self.root.resolve() / "heatmap.tiff"))
        self.assertEqual(artifact.media_type, "image/tiff")
        image_format, values = self.pixels(artifact)
        self.assertEqual(image_format, "uchar")
        np.testing.assert_array_equal(values, 127)
        self.assertEqual(list(self.root.glob(".*.tiff")), [])

    async def test_float32_encoding_publishes_unquantized_scores(self) -> None:
        artifact = await self.run_builder(
            fake_handle(self.config, np.full((32, 32), 0.375, dtype=np.float32)),
            configuration={"encoding": "float32"},
        )
        image_format, values = self.pixels(artifact)
        self.assertEqual(image_format, "float")
        np.testing.assert_allclose(values, 0.375)

    async def test_missing_mask_processes_the_whole_slide(self) -> None:
        handle = fake_handle(self.config, np.full((32, 32), 0.375, dtype=np.float32))
        artifact = await self.run_builder(
            handle, configuration={"encoding": "float32"}, mask=False
        )
        _format, values = self.pixels(artifact)
        np.testing.assert_allclose(values, 0.375)
        # 96x64 at tile 32 and stride 16: 5 x 3 grid positions.
        self.assertEqual(handle.predict.remote.await_count, 15)

    async def test_background_tiles_are_skipped_unless_disabled(self) -> None:
        pixels = np.full((64, 96, 3), 245, dtype=np.uint8)  # white glass
        pixels[:, :48] = (200, 120, 160)  # tissue on the left half
        write_slide(self.root / "slide.tiff", pixels)
        cases = {
            # Columns x = 0, 16, 32 touch the left half, in 3 rows.
            "default": ({"encoding": "float32"}, 9),
            "disabled": ({"encoding": "float32", "skip_background": False}, 15),
        }
        for case, (configuration, calls) in cases.items():
            with self.subTest(case):
                handle = fake_handle(
                    self.config, np.full((32, 32), 0.375, dtype=np.float32)
                )
                artifact = await self.run_builder(
                    handle, configuration=configuration, mask=False
                )
                _format, values = self.pixels(artifact)
                self.assertEqual(handle.predict.remote.await_count, calls)
                scores = values.reshape(64, 96)
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
        artifact = await self.run_builder(handle, configuration={"encoding": "float32"})
        _format, values = self.pixels(artifact)
        np.testing.assert_allclose(values, 0.375)


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
