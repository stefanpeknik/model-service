"""Heatmap builder: one score grid per slide from any `spatial-scores` model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.application import Builder
from model_service.builders.heatmap.declaration import (
    HeatmapConfiguration,
    HeatmapInputs,
    HeatmapOutputs,
)
from model_service.builders.heatmap.processor import (
    HeatmapRequest,
    SlideHeatmapProcessor,
)
from model_service.builders.model_clients import SpatialScoresClient


if TYPE_CHECKING:
    from model_service.builders.application import BuilderContext
    from model_service.builders.settings import WorkerSettings


class HeatmapBuilder(
    Builder[SpatialScoresClient, HeatmapInputs, HeatmapConfiguration, HeatmapOutputs]
):
    def __init__(self, settings: WorkerSettings) -> None:
        """Configure the builder for a replica.

        Args:
            settings: Replica settings for thread and concurrency limits.
        """
        self._settings = settings

    async def run(
        self,
        context: BuilderContext[
            SpatialScoresClient, HeatmapInputs, HeatmapConfiguration
        ],
    ) -> HeatmapOutputs:
        """Write a slide's score grid to an `.npz` artifact.

        Args:
            context: The verified model, the slide and optional region of interest,
                and the configuration.

        Returns:
            The `heatmap` artifact.

        Raises:
            BuilderFailure: A model call failed or returned invalid scores.
            InvalidInvocation: The slide or region of interest is unreadable, or the
                output would replace an input.
        """
        processor = SlideHeatmapProcessor(
            context.model,
            num_threads=self._settings.num_threads,
            max_concurrent_tasks=self._settings.max_concurrent_tasks,
        )
        with context.write_artifact("heatmap") as heatmap:
            await processor.build(
                HeatmapRequest(
                    slide_path=context.inputs.slide,
                    mask_path=context.inputs.region_of_interest,
                    output_path=heatmap.path,
                    stride_fraction=context.configuration.stride_fraction,
                    skip_background=context.configuration.skip_background,
                )
            )
        return HeatmapOutputs(heatmap=heatmap.artifact)
