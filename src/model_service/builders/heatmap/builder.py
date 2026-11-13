"""Heatmap builder: one score grid per slide from any `spatial-scores` model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.base import Builder
from model_service.builders.heatmap.processor import build_heatmap
from model_service.builders.heatmap.spec import (
    HeatmapConfiguration,
    HeatmapInputs,
    HeatmapOutputs,
)
from model_service.deployed_models.model_clients import SpatialScoresClient


if TYPE_CHECKING:
    from model_service.builders.base import BuilderContext


class HeatmapBuilder(
    Builder[SpatialScoresClient, HeatmapInputs, HeatmapConfiguration, HeatmapOutputs]
):
    """Scores a slide tile by tile and writes the score grid."""

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
        with context.write_artifact("heatmap") as (path, heatmap):
            await build_heatmap(
                context.inputs.slide,
                context.inputs.region_of_interest,
                path,
                client=context.model,
                stride_fraction=context.configuration.stride_fraction,
                skip_background=context.configuration.skip_background,
                num_threads=self._settings.num_threads,
                max_concurrent_tasks=self._settings.max_concurrent_tasks,
            )
        return HeatmapOutputs(heatmap=heatmap)
