"""Write one feature row per selected tile to HDF5, in stable grid order."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from builders.contracts import BuilderFailure, FailureCode


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from builders.contracts import TileEmbeddingModel
    from builders.embeddings.predictor import HttpEmbeddingPredictor


async def build_embeddings(
    slide_path: Path,
    mask_path: Path | None,
    output_path: Path,
    *,
    predictor: HttpEmbeddingPredictor,
    model: TileEmbeddingModel,
    stride_fraction: float,
    num_threads: int,
    max_concurrent_tasks: int,
) -> None:
    """Coordinates are tile origins at the slide level closest to the model's mpp."""
    import h5py

    from builders.tile_reader import TileReader, open_slide

    tile_size = model.tile_size
    stride = max(1, round(tile_size * stride_fraction))
    loop = asyncio.get_running_loop()
    tasks: deque[asyncio.Task[tuple[int, int, NDArray[np.float32] | None]]] = deque()
    with (
        open_slide(slide_path, "slide") as slide,
        TileReader.open(slide, mask_path, model.mpp) as reader,
        ThreadPoolExecutor(max_workers=num_threads) as executor,
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

        async def infer(x: int, y: int) -> tuple[int, int, NDArray[np.float32] | None]:
            tile = await loop.run_in_executor(executor, reader.read, x, y, tile_size)
            return x, y, None if tile is None else await predictor.predict(tile)

        async def append_oldest() -> None:
            nonlocal features
            # Keep the task registered until it is joined, including on cancellation.
            x, y, result = await tasks[0]
            tasks.popleft()
            if result is None:
                return
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
            # Include one padded tile even when the slide is smaller than a tile.
            for y in range(0, max(0, height - tile_size) + stride, stride):
                for x in range(0, max(0, width - tile_size) + stride, stride):
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
