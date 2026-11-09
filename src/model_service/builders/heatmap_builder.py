import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from fastapi import FastAPI
from ray import serve


class Config(TypedDict):
    num_threads: int
    max_concurrent_tasks: int


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class HeatmapBuilder:
    num_threads: int
    max_concurrent_tasks: int

    async def reconfigure(self, config: Config) -> None:
        self.num_threads = config["num_threads"]
        self.max_concurrent_tasks = config["max_concurrent_tasks"]

    @fastapi.post("/")
    async def root(
        self,
        model_id: str,
        slide_path: str,
        tissue_mask_path: str,
        output_path: str,
        stride_fraction: float,
        output_bigtiff_tile_height: int,
        output_bigtiff_tile_width: int,
    ) -> str:
        import pyvips
        from ratiopath.masks.mask_builders import MaskBuilder
        from ratiopath.openslide import OpenSlide
        from ratiopath.tiling import grid_tiles

        from model_service.misc.fetch_tissue_tile import fetch_tissue_tile

        model = serve.get_app_handle(model_id)
        model_config = await model.get_config.remote()
        tile_size: int = model_config["tile_size"]
        output_tile_size: int = model_config["output_tile_size"]
        n_channels: int = model_config["n_channels"]
        stride: int = round(stride_fraction * tile_size)

        loop = asyncio.get_running_loop()
        tasks: set[asyncio.Task[Any]] = set()
        with (
            OpenSlide(slide_path) as slide,
            OpenSlide(tissue_mask_path) as tissue_slide,
            ThreadPoolExecutor(max_workers=self.num_threads) as executor,
        ):
            level = slide.closest_level(model_config["mpp"])
            mpp_x, mpp_y = slide.slide_resolution(level)
            extent_x, extent_y = slide.level_dimensions[level]

            tissue_level = tissue_slide.closest_level(model_config["mpp"])
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
                    arr = np.asarray(prediction, dtype=np.float32)

                    if arr.ndim == 2:
                        batch = arr[np.newaxis, np.newaxis, ...]
                    elif arr.ndim == 3:
                        batch = arr[np.newaxis, ...]
                    else:
                        raise ValueError(f"Unexpected prediction shape: {arr.shape}")

                    mask_builder.update_batch(
                        batch=batch,
                        coords=np.array([[y, x]], dtype=np.int64),
                    )

                for x, y in grid_tiles(
                    slide_extent=(extent_x, extent_y),
                    tile_extent=(tile_size, tile_size),
                    stride=(stride, stride),
                ):
                    if len(tasks) >= self.max_concurrent_tasks:
                        done, tasks = await asyncio.wait(
                            tasks, return_when=asyncio.FIRST_COMPLETED
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
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                vips_image.tiffsave(
                    output_path,
                    bigtiff=True,
                    compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
                    tile=True,
                    tile_width=output_bigtiff_tile_width,
                    tile_height=output_bigtiff_tile_height,
                    xres=1000 / mpp_x,
                    yres=1000 / mpp_y,
                    pyramid=True,
                )
            finally:
                mask_builder.cleanup()

        return output_path


app = HeatmapBuilder.bind()  # type: ignore[attr-defined]
