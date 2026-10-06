from pathlib import Path
from typing import TypedDict

from fastapi import FastAPI
from ray import serve

from workflows.heatmap.processor import HeatmapBuildRequest, SlideHeatmapProcessor
from workflows.heatmap.ray_predictor import RayHeatmapPredictor


class Config(TypedDict):
    num_threads: int
    max_concurrent_tasks: int


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class HeatmapBuilder:
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
