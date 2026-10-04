from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Protocol

from pydantic import Field, TypeAdapter


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class HeatmapBuildRequest:
    slide_path: Path
    tissue_mask_path: Path
    output_path: Path
    stride_fraction: float
    output_tile_height: int
    output_tile_width: int


@dataclass(frozen=True, slots=True)
class ModelTileConfiguration:
    tile_size: Annotated[int, Field(strict=True, gt=0)]
    output_tile_size: Annotated[int, Field(strict=True, gt=0)]
    n_channels: Annotated[int, Field(strict=True, gt=0)]
    mpp: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]

    @classmethod
    def from_values(cls, values: Mapping[str, object]) -> ModelTileConfiguration:
        return _MODEL_TILE_CONFIGURATION.validate_python(values)


_MODEL_TILE_CONFIGURATION = TypeAdapter(ModelTileConfiguration)


class HeatmapPredictor(Protocol):
    async def configuration(self) -> ModelTileConfiguration: ...

    async def predict(self, tile: object) -> object: ...


class SlideHeatmapProcessor:
    def __init__(
        self,
        predictor: HeatmapPredictor,
        *,
        num_threads: int,
        max_concurrent_tasks: int,
    ) -> None:
        if num_threads <= 0 or max_concurrent_tasks <= 0:
            raise ValueError("heatmap thread and concurrency limits must be positive")
        self._predictor = predictor
        self._num_threads = num_threads
        self._max_concurrent_tasks = max_concurrent_tasks

    async def build(self, request: HeatmapBuildRequest) -> None:
        import numpy as np
        import pyvips
        from ratiopath.masks.mask_builders import MaskBuilder
        from ratiopath.openslide import OpenSlide
        from ratiopath.tiling import grid_tiles

        from misc.fetch_tissue_tile import fetch_tissue_tile

        model_config = await self._predictor.configuration()
        tile_size = model_config.tile_size
        output_tile_size = model_config.output_tile_size
        n_channels = model_config.n_channels
        target_mpp = model_config.mpp
        stride = max(1, round(request.stride_fraction * tile_size))
        loop = asyncio.get_running_loop()
        tasks: set[asyncio.Task[None]] = set()
        with (
            OpenSlide(str(request.slide_path)) as slide,
            OpenSlide(str(request.tissue_mask_path)) as tissue_slide,
            ThreadPoolExecutor(max_workers=self._num_threads) as executor,
        ):
            level = slide.closest_level(target_mpp)
            mpp_x, mpp_y = slide.slide_resolution(level)
            extent_x, extent_y = slide.level_dimensions[level]
            tissue_level = tissue_slide.closest_level(target_mpp)
            tissue_extent_x, tissue_extent_y = tissue_slide.level_dimensions[
                tissue_level
            ]
            scale_x = tissue_extent_x / extent_x
            scale_y = tissue_extent_y / extent_y
            mask_builder = MaskBuilder(
                source_extents=(extent_y, extent_x),
                source_tile_extent=tile_size,
                output_tile_extent=output_tile_size,
                stride=stride,
                n_channels=n_channels,
                storage="memmap",
            )
            try:

                async def process_tile(x: int, y: int) -> None:
                    tile = await loop.run_in_executor(
                        executor,
                        fetch_tissue_tile,
                        slide,
                        tissue_slide,
                        x,
                        y,
                        level,
                        scale_x,
                        scale_y,
                        tissue_level,
                        tile_size,
                    )
                    if tile is None:
                        return
                    prediction = await self._predictor.predict(tile)
                    batch = np.asarray(prediction, dtype=np.float32).reshape(
                        1,
                        n_channels,
                        output_tile_size,
                        output_tile_size,
                    )
                    mask_builder.update_batch(
                        batch=batch,
                        coords=np.array([[y, x]], dtype=np.int64),
                    )

                for x, y in grid_tiles(
                    slide_extent=(extent_x, extent_y),
                    tile_extent=(tile_size, tile_size),
                    stride=(stride, stride),
                ):
                    if len(tasks) >= self._max_concurrent_tasks:
                        done, tasks = await asyncio.wait(
                            tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        await asyncio.gather(*done)
                    tasks.add(asyncio.create_task(process_tile(x, y)))
                if tasks:
                    await asyncio.gather(*tasks)
                result = np.asarray(mask_builder.finalize()["mask"])
                vips_image = pyvips.Image.new_from_array(result.transpose(1, 2, 0))
                # Preserve tile spacing; resizing to the unpadded extent shifts scores.
                scale_y, scale_x = mask_builder.span / np.asarray(result.shape[1:])
                vips_image = vips_image.resize(
                    float(scale_x), vscale=float(scale_y)
                ).crop(0, 0, extent_x, extent_y)
                vips_image = (vips_image * 255).cast(pyvips.BandFormat.UCHAR)
                vips_image.tiffsave(
                    str(request.output_path),
                    bigtiff=True,
                    compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
                    tile=True,
                    tile_width=request.output_tile_width,
                    tile_height=request.output_tile_height,
                    xres=1000 / mpp_x,
                    yres=1000 / mpp_y,
                    pyramid=True,
                )
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                mask_builder.cleanup()
