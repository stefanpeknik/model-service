"""Build a whole-slide score map from model tiles.

OpenSlide reads run in a thread pool and predictions run concurrently through the
model client. A memory-mapped MaskBuilder averages overlapping scores into a grid
at the model's output resolution, which is written as a score grid.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from model_service.contracts.score_grid import ScoreGrid, write_score_grid


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from model_service.deployed_models.model_clients import SpatialScoresClient


logger = logging.getLogger(__name__)
_PROGRESS_SECONDS = 60


@dataclass(slots=True)
class _StageTiming:
    """Per-tile durations of one stage, to tell slide reads from model waits."""

    count: int = 0
    seconds: float = 0.0

    def add(self, seconds: float) -> None:
        """Record one tile's duration.

        Args:
            seconds: Time the stage took for the tile.
        """
        self.count += 1
        self.seconds += seconds

    def __str__(self) -> str:
        """Format the mean duration per tile for progress logs.

        Returns:
            The mean in milliseconds, or "none" before any tile was timed.
        """
        if not self.count:
            return "none"
        return f"{self.seconds / self.count * 1000:.0f} ms"


async def build_heatmap(
    slide_path: Path,
    mask_path: Path | None,
    output_path: Path,
    *,
    client: SpatialScoresClient,
    stride_fraction: float,
    skip_background: bool,
    num_threads: int,
    max_concurrent_tasks: int,
) -> None:
    """Score a slide's grid tiles and write the averaged score grid.

    Overlapping tile scores are averaged; progress is logged every minute.
    Tile size, output geometry and mpp come from the model's declaration.

    Args:
        slide_path: Slide to tile.
        mask_path: Optional region-of-interest mask limiting which tiles are read.
        output_path: Score grid file to create.
        client: Client of the model that scores each tile.
        stride_fraction: Stride as a fraction of the tile size.
        skip_background: Whether to skip tiles without tissue.
        num_threads: Threads for slide reads.
        max_concurrent_tasks: Tiles in flight at once.

    Raises:
        InvalidInvocation: The slide or mask is not a readable slide.
    """
    import numpy as np
    from ratiopath.masks.mask_builders import MaskBuilder
    from ratiopath.tiling import grid_tiles

    from model_service.builders.slides.tile_reader import TileReader, open_slide

    model = client.declaration
    tile_size = model.tile_size
    output_tile_size = model.output_tile_size
    n_channels = model.n_channels
    stride = max(1, round(stride_fraction * tile_size))
    loop = asyncio.get_running_loop()
    tasks: set[asyncio.Task[None]] = set()
    with (
        open_slide(slide_path, "slide") as slide,
        TileReader.open(
            slide,
            mask_path,
            model.mpp,
            skip_background=skip_background,
        ) as reader,
        ThreadPoolExecutor(max_workers=num_threads) as executor,
    ):
        extent_x, extent_y = slide.level_dimensions[reader.level]
        mpp_x, mpp_y = slide.slide_resolution(reader.level)
        mask_builder = MaskBuilder(
            source_extents=(extent_y, extent_x),
            source_tile_extent=tile_size,
            output_tile_extent=output_tile_size,
            stride=stride,
            n_channels=n_channels,
            storage="memmap",
        )
        logger.info(
            "heatmap of %s: level %d (%dx%d px), %d-px tiles, stride %d, %s",
            model.id,
            reader.level,
            extent_x,
            extent_y,
            tile_size,
            stride,
            "no background filter"
            if reader.tissue is None
            else f"tissue on {reader.tissue.fraction:.0%} of the slide",
        )
        started = last_report = time.monotonic()
        submitted = 0
        reads = _StageTiming()
        predictions = _StageTiming()

        def timed_read(x: int, y: int) -> tuple[NDArray[np.uint8] | None, float]:
            # Timed in the thread, so waiting for a free thread is not counted.
            began = time.perf_counter()
            tile = reader.read(x, y, tile_size)
            return tile, time.perf_counter() - began

        try:

            async def process_tile(x: int, y: int) -> None:
                tile, read_seconds = await loop.run_in_executor(
                    executor, timed_read, x, y
                )
                if tile is None:
                    return
                reads.add(read_seconds)
                began = time.perf_counter()
                prediction = await client.predict(tile)
                predictions.add(time.perf_counter() - began)
                batch = np.asarray(prediction, dtype=np.float32).reshape(
                    1, n_channels, output_tile_size, output_tile_size
                )
                mask_builder.update_batch(
                    batch=batch, coords=np.array([[y, x]], dtype=np.int64)
                )

            for x, y in grid_tiles(
                slide_extent=(max(extent_x, tile_size), max(extent_y, tile_size)),
                tile_extent=(tile_size, tile_size),
                stride=(stride, stride),
            ):
                if len(tasks) >= max_concurrent_tasks:
                    done, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    await asyncio.gather(*done)
                tasks.add(asyncio.create_task(process_tile(x, y)))
                submitted += 1
                if time.monotonic() - last_report >= _PROGRESS_SECONDS:
                    last_report = time.monotonic()
                    logger.info(
                        "heatmap of %s: %d grid positions after %.0f s, at y=%d "
                        "of %d; %d tissue tiles, per tile: read %s, model %s",
                        model.id,
                        submitted,
                        last_report - started,
                        y,
                        extent_y,
                        reads.count,
                        reads,
                        predictions,
                    )
            if tasks:
                await asyncio.gather(*tasks)
            logger.info(
                "heatmap of %s: %d grid positions in %.0f s; %d tissue tiles, "
                "per tile: read %s, model %s",
                model.id,
                submitted,
                time.monotonic() - started,
                reads.count,
                reads,
                predictions,
            )
            finalized = mask_builder.finalize()
            values = np.asarray(finalized["mask"], dtype=np.float32)
            counts = np.asarray(finalized["overlap_counter"])
            # The grid covers the padded span, so a cell is span / cells
            # pixels of the level along each axis.
            scale_y, scale_x = mask_builder.span / np.asarray(values.shape[1:])
            write_score_grid(
                output_path,
                ScoreGrid(
                    values=values,
                    scored=counts.reshape(values.shape[1:]) > 0,
                    level=reader.level,
                    level_size=(extent_x, extent_y),
                    cell_size=float(min(scale_x, scale_y)),
                    mpp=(float(mpp_x), float(mpp_y)),
                ),
            )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            mask_builder.cleanup()
