from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np

from workflow_execution.prostate_heatmap import (
    HeatmapBuildRequest,
    RayProstateHeatmapProcessor,
)


class FakeSlide:
    level_dimensions = ((4, 4),)

    def __init__(self, _path: str) -> None:
        pass

    def __enter__(self) -> FakeSlide:
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

    def cast(self, _format: object) -> FakeVipsImage:
        return self

    def tiffsave(self, path: str, **options: object) -> None:
        self.recorded["tiff_options"] = options
        Path(path).write_bytes(b"tiff")


class FakeMaskBuilder:
    def __init__(self, recorded: dict[str, Any], **options: object) -> None:
        self.recorded = recorded
        recorded["mask_options"] = options

    def update_batch(self, *, batch: np.ndarray, coords: np.ndarray) -> None:
        self.recorded["batch"] = batch
        self.recorded["coords"] = coords

    def finalize(self) -> dict[str, np.ndarray]:
        return {"mask": np.zeros((4, 4), dtype=np.float32)}

    def resize_to_source(self, _result: np.ndarray) -> FakeVipsImage:
        return FakeVipsImage(self.recorded)

    def cleanup(self) -> None:
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
            predict=SimpleNamespace(remote=AsyncMock(return_value=[0.75])),
        )

        def grid_tiles(**options: object) -> list[tuple[int, int]]:
            recorded["grid_options"] = options
            return [(0, 0)]

        fake_modules = {
            "pyvips": module(
                "pyvips",
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
            "ray": module(
                "ray", serve=SimpleNamespace(get_app_handle=lambda _name: model)
            ),
            "misc.fetch_tissue_tile": module(
                "misc.fetch_tissue_tile",
                fetch_tissue_tile=lambda *_args: np.zeros((4, 4, 3)),
            ),
        }
        with TemporaryDirectory() as directory, patch.dict(sys.modules, fake_modules):
            output = Path(directory) / "heatmap.tiff"
            await RayProstateHeatmapProcessor(
                num_threads=1, max_concurrent_tasks=1
            ).build(
                HeatmapBuildRequest(
                    slide_path=Path(directory) / "slide.svs",
                    tissue_mask_path=Path(directory) / "roi.tiff",
                    output_path=output,
                    stride_fraction=0.01,
                    output_tile_height=128,
                    output_tile_width=256,
                )
            )
            self.assertEqual(output.read_bytes(), b"tiff")

        self.assertEqual(recorded["grid_options"]["stride"], (1, 1))
        self.assertEqual(recorded["mask_options"]["stride"], 1)
        np.testing.assert_array_equal(recorded["batch"], [[[[0.75]]]])
        np.testing.assert_array_equal(recorded["coords"], [[0, 0]])
        self.assertEqual(recorded["tiff_options"]["tile_width"], 256)
        self.assertEqual(recorded["tiff_options"]["tile_height"], 128)
        self.assertEqual(recorded["tiff_options"]["xres"], 2000)
        self.assertTrue(recorded["cleaned"])
