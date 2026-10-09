"""Read bounded tile batches and append deterministic feature rows to HDF5."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from workflow_execution.contracts import WorkflowFailure


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from workflows.embeddings.predictor import EmbeddingModel, HttpEmbeddingPredictor
    from workflows.settings import WorkflowWorkerSettings


async def build_embeddings(
    slide_path: Path,
    mask_path: Path,
    output_path: Path,
    *,
    predictor: HttpEmbeddingPredictor,
    model: EmbeddingModel,
    stride_fraction: float,
    settings: WorkflowWorkerSettings,
) -> None:
    """Write selected tile features incrementally, joining all tasks before closing readers.

    Args:
        slide_path: OpenSlide-readable whole-slide input.
        mask_path: Nonzero TIFF mask pixels selecting intersecting tiles.
        output_path: Temporary artifact path owned by the surrounding workflow.
        predictor: Invocation-owned raw-model HTTP adapter.
        model: Input tile geometry and expected summary order.
        stride_fraction: Grid step as a fraction of model tile size.
        settings: Deployment-only thread and outstanding-tile limits.

    Returns:
        None: A closed HDF5 file containing features, coordinates, and sampling metadata.

    Raises:
        WorkflowFailure: Raw prediction fails or feature dimensions vary between tiles.
        OSError: Input access or output writing fails.
        asyncio.CancelledError: Cancellation propagates after joining tile tasks.
    """
    import h5py
    from ratiopath.openslide import OpenSlide

    from workflows.heatmap.tile_reader import fetch_tissue_tile, select_mask_level

    stride = max(1, round(model.tile_size * stride_fraction))
    loop = asyncio.get_running_loop()
    tasks: deque[asyncio.Task[tuple[int, int, NDArray[np.float32] | None]]] = deque()
    with (
        OpenSlide(str(slide_path)) as slide,
        OpenSlide(str(mask_path)) as mask,
        ThreadPoolExecutor(max_workers=settings.num_threads) as executor,
        h5py.File(output_path, "w") as output,
    ):
        width, height = slide.dimensions
        mpp_x, mpp_y = slide.slide_resolution(0)
        mask_level = select_mask_level(mask, (mpp_x, mpp_y))
        mask_width, mask_height = mask.level_dimensions[mask_level]
        output.attrs.update(
            {
                "schema_version": 1,
                "model_id": model.identifier,
                "summary_names": model.summaries,
                "coordinate_space": "slide_level_0",
                "slide_width": width,
                "slide_height": height,
                "slide_level": 0,
                "mpp_x": mpp_x,
                "mpp_y": mpp_y,
                "tile_size": model.tile_size,
                "stride": stride,
            }
        )
        coordinates = output.create_dataset(
            "coordinates", shape=(0, 2), maxshape=(None, 2), dtype="int64"
        )
        features = None

        async def infer(x: int, y: int) -> tuple[int, int, NDArray[np.float32] | None]:
            """Read one level-zero tile and skip inference when its mask is empty.

            Args:
                x: Horizontal tile origin in original slide pixels.
                y: Vertical tile origin in original slide pixels.

            Returns:
                tuple: Original x/y coordinates and features, or None for an empty mask.

            Raises:
                WorkflowFailure: Raw inference fails or returns invalid features.
            """
            tile = await loop.run_in_executor(
                executor,
                fetch_tissue_tile,
                slide,
                mask,
                x,
                y,
                0,
                mask_width / width,
                mask_height / height,
                mask_level,
                model.tile_size,
            )
            return x, y, None if tile is None else await predictor.predict(tile)

        async def append_oldest() -> None:
            """Join the oldest task and append its selected tile in stable grid order.

            Returns:
                None: A nonempty result is appended to coordinates and feature datasets.

            Raises:
                WorkflowFailure: Prediction fails or feature dimensions change.
            """
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
                raise WorkflowFailure(
                    code="invalid_model_output",
                    safe_message="embedding feature dimensions changed between tiles",
                    retryable=False,
                )
            count = len(coordinates)
            coordinates.resize(count + 1, axis=0)
            features.resize(count + 1, axis=0)
            coordinates[count] = (x, y)
            features[count] = result

        try:
            # Include one padded tile even when the slide is smaller than the model tile.
            for y in range(0, max(0, height - model.tile_size) + stride, stride):
                for x in range(0, max(0, width - model.tile_size) + stride, stride):
                    if len(tasks) >= settings.max_concurrent_tasks:
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
