from __future__ import annotations

import asyncio
import gc
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, patch

import numpy as np

from workflow_execution.heatmap import (
    HeatmapBuildRequest,
    ModelTileConfiguration,
    SlideHeatmapProcessor,
)
from workflow_execution.ray_heatmap import RayHeatmapPredictor


class FakeSlide:
    level_dimensions = ((4, 4),)

    def __init__(self, _path: str) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def closest_level(self, _mpp: float) -> int:
        return 0

    def slide_resolution(self, _level: int) -> tuple[float, float]:
        return 0.5, 0.5


class FakeVipsImage:
    def __init__(self, recorded: dict[str, Any]) -> None:
        self.recorded = recorded

    def __mul__(self, _value: int) -> FakeVipsImage:
        return self

    def resize(self, _scale: float, *, vscale: float) -> FakeVipsImage:
        return self

    def crop(self, _x: int, _y: int, _width: int, _height: int) -> FakeVipsImage:
        return self

    def cast(self, _format: object) -> FakeVipsImage:
        return self

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


class ProstateHeatmapProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_valid_stride_still_processes_and_writes_tiff(self) -> None:
        recorded: dict[str, Any] = {}
        model = SimpleNamespace(
            configuration=AsyncMock(return_value=ModelTileConfiguration(4, 1, 1, 0.5)),
            predict=AsyncMock(return_value=0.75),
        )

        def grid_tiles(**options: object) -> list[tuple[int, int]]:
            recorded["grid_options"] = options
            return [(0, 0)]

        fake_modules = {
            "pyvips": module(
                "pyvips",
                Image=SimpleNamespace(
                    new_from_array=lambda _value: FakeVipsImage(recorded)
                ),
                BandFormat=SimpleNamespace(UCHAR="uchar"),
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
            "misc.fetch_tissue_tile": module(
                "misc.fetch_tissue_tile",
                fetch_tissue_tile=lambda *_args: np.zeros((4, 4, 3)),
            ),
        }
        with TemporaryDirectory() as directory, patch.dict(sys.modules, fake_modules):
            output = Path(directory) / "heatmap.tiff"
            request = HeatmapBuildRequest(
                slide_path=Path(directory) / "slide.svs",
                tissue_mask_path=Path(directory) / "roi.tiff",
                output_path=output,
                stride_fraction=0.01,
                output_tile_height=128,
                output_tile_width=256,
            )
            processor = SlideHeatmapProcessor(
                model, num_threads=1, max_concurrent_tasks=2
            )
            await processor.build(request)
            self.assertEqual(output.read_bytes(), b"tiff")
            np.testing.assert_array_equal(recorded["batch"], [[[[0.75]]]])

            model.configuration.return_value = ModelTileConfiguration(4, 2, 1, 0.5)
            model.predict.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            await processor.build(request)
            np.testing.assert_allclose(recorded["batch"], [[[[0.1, 0.2], [0.3, 0.4]]]])

            model.predict.return_value = [0.75]
            with self.assertRaises(ValueError):
                await processor.build(request)
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

            fake_modules["ratiopath.tiling"].grid_tiles = lambda **_options: [
                (0, 0),
                (1, 0),
                (2, 0),
            ]
            model.predict.side_effect = failing_prediction
            recorded["check_cleanup"] = lambda: self.assertTrue(cancelled.is_set())
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                await processor.build(request)
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

            model.predict.side_effect = simultaneous_failures
            loop = asyncio.get_running_loop()
            previous_handler = loop.get_exception_handler()
            unhandled = []
            loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
            try:
                with self.assertRaisesRegex(RuntimeError, "simultaneous failure"):
                    await processor.build(request)
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
        self.assertTrue(recorded["cleaned"])

    async def test_ray_adapter_preserves_model_id_and_tile(self) -> None:
        model = SimpleNamespace(
            get_config=SimpleNamespace(
                remote=AsyncMock(
                    return_value={
                        "tile_size": 4,
                        "output_tile_size": 1,
                        "n_channels": 1,
                        "mpp": 0.5,
                    }
                )
            ),
            predict=SimpleNamespace(remote=AsyncMock(return_value=0.75)),
        )
        get_handle = unittest.mock.Mock(return_value=model)
        with patch.dict(
            sys.modules,
            {"ray": module("ray", serve=SimpleNamespace(get_app_handle=get_handle))},
        ):
            predictor = RayHeatmapPredictor("chosen-model")
            get_handle.assert_not_called()
            self.assertEqual(
                await predictor.configuration(), ModelTileConfiguration(4, 1, 1, 0.5)
            )
            tile = np.zeros((3, 4, 4), dtype=np.uint8)
            self.assertEqual(await predictor.predict(tile), 0.75)
        get_handle.assert_called_once_with("chosen-model")
        self.assertIs(model.predict.remote.call_args.args[0], tile)

    def test_model_metadata_rejects_invalid_dimensions_and_resolution(self) -> None:
        valid = {"tile_size": 4, "output_tile_size": 1, "n_channels": 1, "mpp": 0.5}
        for name, value in (
            ("tile_size", True),
            ("output_tile_size", 0),
            ("n_channels", "1"),
            ("mpp", float("nan")),
            ("mpp", float("inf")),
            ("mpp", 0),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                ModelTileConfiguration.from_values(valid | {name: value})
