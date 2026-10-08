from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    SharedArtifact,
    WorkflowResult,
)
from workflows.artifacts import atomic_artifact
from workflows.heatmap.processor import HeatmapBuildRequest, ScoreEncoding
from workflows.heatmap.prostate import ProstateConfiguration


if TYPE_CHECKING:
    from collections.abc import Mapping

    from workflow_execution.application import WorkflowContext
    from workflows.heatmap.prostate import HeatmapProcessor


class SegmentationConfiguration(ProstateConfiguration):
    output_path: str = "segmentation.tiff"


def validate_configuration(value: Mapping[str, object]) -> SegmentationConfiguration:
    try:
        config = SegmentationConfiguration.model_validate(value)
        if config.model_fields_set != SegmentationConfiguration.model_fields.keys():
            raise ValueError("incomplete configuration")
        return config
    except (ValueError, ValidationError) as error:
        raise InvalidWorkflowInvocation("invalid segmentation configuration") from error


class EPISegWorkflow:
    def __init__(self, processor: HeatmapProcessor) -> None:
        self._processor = processor

    async def run(self, context: WorkflowContext) -> WorkflowResult:
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
