"""Heatmap builder: one score grid per slide from any `spatial-scores` model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.application import Builder
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


class HeatmapBuilder(Builder[SpatialScoresModel, HeatmapConfiguration]):
    def __init__(self, settings: WorkerSettings) -> None:
        """Configure the builder for a replica.

        Args:
            settings: Replica settings for thread and concurrency limits.
        """
        self._settings = settings

    async def run(
        self, context: BuilderContext[SpatialScoresModel, HeatmapConfiguration]
    ) -> BuilderResult:
        """Write a slide's score grid to an `.npz` artifact.

        Checks the running model against its catalog entry first, then writes
        through a temporary file so a failed run leaves no partial artifact.

        Args:
            context: The run's inputs, model and configuration.

        Returns:
            The `heatmap` artifact.

        Raises:
            BuilderFailure: The model's configuration is invalid or differs from its
                catalog entry.
            InvalidInvocation: The slide or region of interest is unreadable, or the
                output would replace an input.
        """
        model = context.model
        config = context.configuration
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
                    skip_background=config.skip_background,
                ),
            )
        return BuilderResult(outputs={"heatmap": SharedArtifact(path=str(destination))})
