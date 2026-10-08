"""Expose the older direct heatmap-building endpoint through Ray Serve.

This entry point invokes the shared processor using caller-supplied paths and model IDs.
It bypasses durable admission and the workflow wrapper that atomically publishes
completed artifacts.
"""

from pathlib import Path
from typing import TypedDict

from fastapi import FastAPI
from ray import serve

from workflows.heatmap.processor import HeatmapBuildRequest, SlideHeatmapProcessor
from workflows.heatmap.ray_predictor import RayHeatmapPredictor


class Config(TypedDict):
    """Declare read-thread and in-flight tile limits for the legacy builder deployment.

    Attributes:
        num_threads: Number of synchronous slide-read threads.
        max_concurrent_tasks: Maximum in-flight tile read/prediction tasks.
    """

    num_threads: int
    max_concurrent_tasks: int


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class HeatmapBuilder:
    """Build slide heatmaps directly from HTTP parameters using a chosen Ray classifier."""

    async def reconfigure(self, config: Config) -> None:
        """Store the read-thread and tile-concurrency limits supplied by Serve user configuration.

        Args:
            config: Ray Serve user configuration for read-thread and tile-concurrency
                limits.

        Returns:
            None: No return value.
        """
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
        """Create the output directory, invoke the common slide processor, and return the supplied destination path.

        Args:
            model_id: Ray Serve classifier application identifier.
            slide_path: Shared slide file path supplied to the legacy builder.
            tissue_mask_path: Shared region-mask file path supplied to the legacy
                builder.
            output_path: Destination artifact path or configured relative output path.
            stride_fraction: Input tile step as a fraction of tile extent.
            output_bigtiff_tile_height: TIFF storage tile height in pixels for the
                legacy builder.
            output_bigtiff_tile_width: TIFF storage tile width in pixels for the legacy
                builder.

        Returns:
            str: Supplied destination path after the processor has written its TIFF.
        """
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        processor = SlideHeatmapProcessor(
            RayHeatmapPredictor(model_id),
            num_threads=self.num_threads,
            max_concurrent_tasks=self.max_concurrent_tasks,
        )
        await processor.build(
            HeatmapBuildRequest(
                slide_path=Path(slide_path),
                tissue_mask_path=Path(tissue_mask_path),
                output_path=destination,
                stride_fraction=stride_fraction,
                output_tile_height=output_bigtiff_tile_height,
                output_tile_width=output_bigtiff_tile_width,
            )
        )
        return output_path


app = HeatmapBuilder.bind()  # type: ignore[attr-defined]
