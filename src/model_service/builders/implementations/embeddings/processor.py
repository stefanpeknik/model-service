"""Write one feature row per selected tile to HDF5, in stable grid order."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from model_service.contracts.failures import BuilderFailure, FailureCode


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from model_service.builders.model_clients.tile_embedding import TileEmbeddingClient


async def build_embeddings(
    slide_path: Path,
    mask_path: Path | None,
    output_path: Path,
    *,
    client: TileEmbeddingClient,
    stride_fraction: float,
    skip_background: bool,
    min_tissue_fraction: float,
    num_threads: int,
    max_concurrent_tasks: int,
    report_progress: Callable[[float], None],
) -> None:
    """Embed a slide's grid tiles and write them to an HDF5 file.

    Coordinates are tile origins at the slide level closest to the model's mpp.
    Tiles the mask or the tissue filter excludes get no row; rows follow grid
    order.

    Args:
        slide_path: Slide to tile.
        mask_path: Optional region-of-interest mask limiting which tiles are read.
        output_path: HDF5 file to create.
        client: Client of the model that embeds each tile.
        stride_fraction: Stride as a fraction of the tile size.
        skip_background: Whether to skip tiles without tissue.
        min_tissue_fraction: Share of a tile that must be tissue to be embedded.
        num_threads: Threads for slide reads.
        max_concurrent_tasks: Tiles in flight at once.
        report_progress: Takes the share of grid positions done.

    Raises:
        InvalidInvocation: The slide or mask is not a readable slide, or the
            slide has no level coarse enough for tissue detection.
        BuilderFailure: Feature dimensions change between tiles, or the model
            fails.
    """
    import h5py
    from ratiopath.tiling import grid_tiles

    from model_service.slides.tile_reader import TileReader, open_slide

    model = client.declaration
    tile_size = model.tile_size
    stride = max(1, round(tile_size * stride_fraction))
    loop = asyncio.get_running_loop()
    tasks: deque[asyncio.Task[tuple[int, int, NDArray[np.float32] | None]]] = deque()
    with (
        open_slide(slide_path, "slide") as slide,
        TileReader.open(
            slide,
            mask_path,
            model.mpp,
            skip_background=skip_background,
            min_tissue_fraction=min_tissue_fraction,
        ) as reader,
        ThreadPoolExecutor(max_workers=num_threads) as executor,
        # One thread writes the table, in the order rows are handed to it.
        ThreadPoolExecutor(max_workers=1) as writer,
        h5py.File(output_path, "w") as output,
    ):
        level = reader.level
        width, height = slide.level_dimensions[level]
        mpp_x, mpp_y = slide.slide_resolution(level)
        output.attrs.update(
            {
                "schema_version": 2,
                "model_id": model.id,
                "summary_names": model.summaries,
                "coordinate_space": "slide_level",
                "slide_level": level,
                "level_downsample": slide.level_downsamples[level],
                "slide_width": width,
                "slide_height": height,
                "mpp_x": mpp_x,
                "mpp_y": mpp_y,
                "model_mpp": model.mpp,
                "tile_size": tile_size,
                "stride": stride,
            }
        )
        coordinates = output.create_dataset(
            "coordinates", shape=(0, 2), maxshape=(None, 2), dtype="int64"
        )
        features = None
        finished = 0
        positions = list(
            grid_tiles(
                slide_extent=(max(width, tile_size), max(height, tile_size)),
                tile_extent=(tile_size, tile_size),
                stride=(stride, stride),
            )
        )

        async def infer(x: int, y: int) -> tuple[int, int, NDArray[np.float32] | None]:
            """Read and embed one tile.

            Args:
                x: Tile x on the read level.
                y: Tile y on the read level.

            Returns:
                The tile position and its features, or None when it is skipped.
            """
            tile = await loop.run_in_executor(executor, reader.read, x, y, tile_size)
            return x, y, None if tile is None else await client.embed(tile)

        async def append_oldest() -> None:
            """Wait for the oldest tile and append its row to the table.

            Raises:
                BuilderFailure: The features differ in shape from earlier tiles.
            """
            nonlocal finished
            # Keep the task registered until it is joined, including on cancellation.
            x, y, result = await tasks[0]
            tasks.popleft()
            finished += 1
            report_progress(finished / len(positions))
            if result is None:
                return
            await loop.run_in_executor(writer, append_row, x, y, result)

        def append_row(x: int, y: int, result: NDArray[np.float32]) -> None:
            """Append one tile's row to the table; runs on the writer thread.

            Args:
                x: Tile x on the read level.
                y: Tile y on the read level.
                result: The tile's features.

            Raises:
                BuilderFailure: The features differ in shape from earlier tiles.
            """
            nonlocal features
            if features is None:
                features = output.create_dataset(
                    "features",
                    shape=(0, *result.shape),
                    maxshape=(None, *result.shape),
                    dtype="float32",
                )
            if features.shape[1:] != result.shape:
                raise BuilderFailure(
                    code=FailureCode.INVALID_MODEL_OUTPUT,
                    safe_message="embedding feature dimensions changed between tiles",
                    retryable=False,
                )
            count = len(coordinates)
            coordinates.resize(count + 1, axis=0)
            features.resize(count + 1, axis=0)
            coordinates[count] = (x, y)
            features[count] = result

        try:
            for x, y in positions:
                if len(tasks) >= max_concurrent_tasks:
                    await append_oldest()
                tasks.append(asyncio.create_task(infer(x, y)))
            while tasks:
                await append_oldest()
            if features is None:
                shape = (
                    (0, 0)
                    if len(model.summaries) == 1
                    else (0, len(model.summaries), 0)
                )
                output.create_dataset("features", shape=shape, dtype="float32")
            output.attrs["tile_count"] = len(coordinates)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
