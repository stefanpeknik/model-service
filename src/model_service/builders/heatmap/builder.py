"""Heatmap builder: one score map per slide from any `spatial-scores` model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.artifacts import atomic_artifact
from model_service.builders.contracts import (
    BuilderResult,
    SharedArtifact,
    SpatialScoresModel,
)
from model_service.builders.heatmap.predictor import RayModelPredictor
from model_service.builders.heatmap.processor import (
    HeatmapRequest,
    SlideHeatmapProcessor,
)
from model_service.builders.registry import HeatmapConfiguration


if TYPE_CHECKING:
    from model_service.builders.application import BuilderContext
    from model_service.builders.settings import WorkerSettings


class HeatmapBuilder:
    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings

    async def run(self, context: BuilderContext) -> BuilderResult:
        model = context.model
        config = context.configuration
        assert isinstance(model, SpatialScoresModel)
        assert isinstance(config, HeatmapConfiguration)
        predictor = RayModelPredictor(model.id)
        await predictor.verify(model)
        processor = SlideHeatmapProcessor(
            predictor,
            num_threads=self._settings.num_threads,
            max_concurrent_tasks=self._settings.max_concurrent_tasks,
        )
        slide = context.input_path("slide")
        mask = context.optional_input_path("region_of_interest")
        destination = context.output_path("heatmap")
        inputs = (slide,) if mask is None else (slide, mask)
        with atomic_artifact(destination, inputs=inputs) as temporary:
            await processor.build(
                model,
                HeatmapRequest(
                    slide_path=slide,
                    mask_path=mask,
                    output_path=temporary,
                    stride_fraction=config.stride_fraction,
                    output_tile_height=config.output_tile_height,
                    output_tile_width=config.output_tile_width,
                    encoding=config.encoding,
                    skip_background=config.skip_background,
                ),
            )
        return BuilderResult(
            outputs={
                "heatmap": SharedArtifact(
                    path=str(destination), media_type="image/tiff"
                )
            }
        )
