"""Produce a floating-point EPISeg score map without changing raw inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from workflow_execution.contracts import (
    SharedArtifact,
    WorkflowResult,
)
from workflows.artifacts import atomic_artifact
from workflows.configuration import validate_resolved_configuration
from workflows.heatmap.configuration import ScoreMapConfiguration
from workflows.heatmap.processor import HeatmapBuildRequest, ScoreEncoding


if TYPE_CHECKING:
    from collections.abc import Mapping

    from workflow_execution.application import WorkflowContext
    from workflows.heatmap.prostate import HeatmapProcessor


class SegmentationConfiguration(ScoreMapConfiguration):
    """Use shared score-map geometry with a segmentation-specific default output path."""

    output_path: str = Field(default="segmentation.tiff", min_length=1)


def validate_configuration(value: Mapping[str, object]) -> SegmentationConfiguration:
    """Require complete EPISeg output and stride settings.

    Args:
        value: Fully resolved execution settings from the catalog.

    Returns:
        SegmentationConfiguration: Validated immutable score-map settings.

    Raises:
        InvalidWorkflowInvocation: Settings are invalid or incomplete.
    """
    return validate_resolved_configuration(
        SegmentationConfiguration, value, workflow_name="EPISeg segmentation"
    )


class EPISegWorkflow:
    """Build and atomically publish a stitched float32 segmentation score TIFF."""

    def __init__(self, processor: HeatmapProcessor) -> None:
        """Retain the injected spatial processor without acquiring model resources.

        Args:
            processor: Shared tiling/reconstruction implementation wired to EPISeg.
        """
        self._processor = processor

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """Write float scores and return the completed TIFF artifact.

        Args:
            context: Invocation-owned input/output access and resolved configuration.

        Returns:
            WorkflowResult: One completed segmentation artifact with TIFF media type.

        Raises:
            InvalidWorkflowInvocation: Configuration or output/input aliasing is invalid.
            OSError: Reading, writing, or publishing an artifact fails.
            RuntimeError: Computation produces no regular artifact file.
            asyncio.CancelledError: Cancellation propagates after temporary-file cleanup.
        """
        config = validate_configuration(context.configuration)
        slide = context.input_path("slide")
        mask = context.input_path("region_of_interest")
        destination = context.output_path(config.output_path)
        with atomic_artifact(destination, inputs=(slide, mask)) as temporary:
            await self._processor.build(
                HeatmapBuildRequest(
                    slide_path=slide,
                    tissue_mask_path=mask,
                    output_path=temporary,
                    stride_fraction=config.stride_fraction,
                    output_tile_height=config.output_tile_height,
                    output_tile_width=config.output_tile_width,
                    encoding=ScoreEncoding.FLOAT32,
                )
            )
        return WorkflowResult(
            outputs={
                "segmentation": SharedArtifact(
                    path=str(destination),
                    media_type="image/tiff",
                )
            }
        )
