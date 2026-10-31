"""Build a whole-slide score map from model tiles.

OpenSlide reads run in a thread pool and predictions run concurrently through the
predictor. A memory-mapped MaskBuilder averages overlapping scores into a grid
at the model's output resolution, which is written as a score grid.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from model_service.builders.score_grid import ScoreGrid, write_score_grid


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.builders.contracts import SpatialScoresModel


logger = logging.getLogger(__name__)
_PROGRESS_SECONDS = 60


@dataclass(frozen=True, slots=True)
class HeatmapRequest:
    slide_path: Path
    mask_path: Path | None
    output_path: Path
    stride_fraction: float
    skip_background: bool = False


@dataclass(slots=True)
class _StageTiming:
    """Per-tile durations of one stage, to tell slide reads from model waits."""

    count: int = 0
    seconds: float = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self.seconds += seconds

    def __str__(self) -> str:
        if not self.count:
            return "none"
        return f"{self.seconds / self.count * 1000:.0f} ms"


class TilePredictor(Protocol):
    async def predict(self, tile: object) -> object:
        """Return `n_channels x output_tile_size²` scores for one CHW uint8 tile."""
        ...


class SlideHeatmapProcessor:
    def __init__(
        self, predictor: TilePredictor, *, num_threads: int, max_concurrent_tasks: int
    ) -> None:
        if num_threads <= 0 or max_concurrent_tasks <= 0:
            raise ValueError("heatmap thread and concurrency limits must be positive")
        self._predictor = predictor
        self._num_threads = num_threads
        self._max_concurrent_tasks = max_concurrent_tasks

    async def build(self, model: SpatialScoresModel, request: HeatmapRequest) -> None:
        import numpy as np
        from ratiopath.masks.mask_builders import MaskBuilder
        from ratiopath.tiling import grid_tiles

        from model_service.builders.tile_reader import TileReader, open_slide

        tile_size = model.tile_size
        output_tile_size = model.output_tile_size
        n_channels = model.n_channels
        stride = max(1, round(request.stride_fraction * tile_size))
        loop = asyncio.get_running_loop()
        tasks: set[asyncio.Task[None]] = set()
        with (
            open_slide(request.slide_path, "slide") as slide,
            TileReader.open(
                slide,
                request.mask_path,
                model.mpp,
                skip_background=request.skip_background,
            ) as reader,
            ThreadPoolExecutor(max_workers=self._num_threads) as executor,
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

            def timed_read(x: int, y: int) -> tuple[object, float]:
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
                    prediction = await self._predictor.predict(tile)
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
                    if len(tasks) >= self._max_concurrent_tasks:
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
                    request.output_path,
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
