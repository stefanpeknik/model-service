from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    SharedArtifact,
    WorkflowResult,
)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from workflow_execution.application import WorkflowContext


@dataclass(frozen=True, slots=True)
class HeatmapBuildRequest:
    slide_path: Path
    tissue_mask_path: Path
    output_path: Path
    stride_fraction: float
    output_tile_height: int
    output_tile_width: int


class HeatmapProcessor(Protocol):
    async def build(self, request: HeatmapBuildRequest) -> None: ...


class ProstateHeatmapWorkflow:
    def __init__(self, processor: HeatmapProcessor) -> None:
        self._processor = processor

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        configuration = validate_configuration(context.configuration)
        final_path = context.output_path(configuration.output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = final_path.with_name(
            f".{final_path.stem}.{uuid4().hex}{final_path.suffix}"
        )
        try:
            await self._processor.build(
                HeatmapBuildRequest(
                    slide_path=context.input_path("slide"),
                    tissue_mask_path=context.input_path("region_of_interest"),
                    output_path=temporary_path,
                    stride_fraction=configuration.stride_fraction,
                    output_tile_height=configuration.output_tile_height,
                    output_tile_width=configuration.output_tile_width,
                )
            )
            if not temporary_path.is_file():
                raise RuntimeError("heatmap processor did not produce a regular file")
            os.replace(temporary_path, final_path)
            if not final_path.is_file():
                raise RuntimeError("completed heatmap is not a regular file")
        finally:
            temporary_path.unlink(missing_ok=True)
        return WorkflowResult(
            outputs={
                "heatmap": SharedArtifact(
                    path=str(final_path),
                    media_type="image/tiff",
                )
            }
        )


@dataclass(frozen=True, slots=True)
class ProstateConfiguration:
    output_path: str
    stride_fraction: float
    output_tile_height: int
    output_tile_width: int


def validate_configuration(value: Mapping[str, object]) -> ProstateConfiguration:
    expected = {
        "output_path",
        "stride_fraction",
        "output_tile_height",
        "output_tile_width",
    }
    if set(value) != expected:
        raise InvalidWorkflowInvocation(
            "resolved configuration does not match prostate heatmap workflow"
        )
    output_path = value["output_path"]
    stride_fraction = value["stride_fraction"]
    output_tile_height = value["output_tile_height"]
    output_tile_width = value["output_tile_width"]
    if not isinstance(output_path, str) or not output_path:
        raise InvalidWorkflowInvocation("output_path must be a non-empty string")
    if (
        not isinstance(stride_fraction, int | float)
        or isinstance(stride_fraction, bool)
        or not 0 < stride_fraction <= 1
    ):
        raise InvalidWorkflowInvocation("stride_fraction must be between zero and one")
    if (
        not isinstance(output_tile_height, int)
        or isinstance(output_tile_height, bool)
        or output_tile_height <= 0
        or not isinstance(output_tile_width, int)
        or isinstance(output_tile_width, bool)
        or output_tile_width <= 0
    ):
        raise InvalidWorkflowInvocation(
            "output tile dimensions must be positive integers"
        )
    return ProstateConfiguration(
        output_path=output_path,
        stride_fraction=float(stride_fraction),
        output_tile_height=output_tile_height,
        output_tile_width=output_tile_width,
    )


class RayProstateHeatmapProcessor:
    def __init__(self, *, num_threads: int, max_concurrent_tasks: int) -> None:
        self._num_threads = num_threads
        self._max_concurrent_tasks = max_concurrent_tasks

    async def build(self, request: HeatmapBuildRequest) -> None:
        import numpy as np
        import pyvips
        from ratiopath.masks.mask_builders import MaskBuilder
        from ratiopath.openslide import OpenSlide
        from ratiopath.tiling import grid_tiles
        from ray import serve

        from misc.fetch_tissue_tile import fetch_tissue_tile

        model = serve.get_app_handle("prostate-classifier-1")
        model_config = await model.get_config.remote()
        tile_size = int(model_config["tile_size"])
        output_tile_size = int(model_config["output_tile_size"])
        n_channels = int(model_config["n_channels"])
        target_mpp = float(model_config["mpp"])
        stride = round(request.stride_fraction * tile_size)
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
                    prediction = await model.predict.remote(tile)
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
                        for task in done:
                            task.result()
                    tasks.add(asyncio.create_task(process_tile(x, y)))
                if tasks:
                    done, _ = await asyncio.wait(tasks)
                    for task in done:
                        task.result()
                result = np.asarray(mask_builder.finalize()["mask"])
                vips_image = mask_builder.resize_to_source(result)
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
