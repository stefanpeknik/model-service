from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    SharedArtifact,
    WorkflowResult,
)
from workflows.heatmap.processor import HeatmapBuildRequest


if TYPE_CHECKING:
    from collections.abc import Mapping

    from workflow_execution.application import WorkflowContext
    from workflow_execution.contracts import JsonValue


class HeatmapProcessor(Protocol):
    async def build(self, request: HeatmapBuildRequest) -> None: ...


class ProstateHeatmapWorkflow:
    def __init__(self, processor: HeatmapProcessor) -> None:
        self._processor = processor

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        configuration = validate_configuration(context.configuration)
        final_path = context.output_path(configuration.output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = final_path.with_name(
            f".{final_path.stem}.{uuid4().hex}{final_path.suffix}"
        )
        try:
            await self._processor.build(
                HeatmapBuildRequest(
                    slide_path=context.input_path("slide"),
                    tissue_mask_path=context.input_path("region_of_interest"),
                    output_path=temporary_path,
                    stride_fraction=configuration.stride_fraction,
                    output_tile_height=configuration.output_tile_height,
                    output_tile_width=configuration.output_tile_width,
                )
            )
            if not temporary_path.is_file():
                raise RuntimeError("heatmap processor did not produce a regular file")
            os.replace(temporary_path, final_path)
            if not final_path.is_file():
                raise RuntimeError("completed heatmap is not a regular file")
        finally:
            temporary_path.unlink(missing_ok=True)
        return WorkflowResult(
            outputs={
                "heatmap": SharedArtifact(
                    path=str(final_path),
                    media_type="image/tiff",
                )
            }
        )


class ProstateConfiguration(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    output_path: str = Field(default="heatmap.tiff", min_length=1)
    stride_fraction: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    output_tile_height: int = Field(default=256, gt=0)
    output_tile_width: int = Field(default=256, gt=0)

    def as_values(self) -> dict[str, JsonValue]:
        return self.model_dump()


def default_configuration() -> dict[str, JsonValue]:
    return ProstateConfiguration().as_values()


def validate_configuration(value: Mapping[str, object]) -> ProstateConfiguration:
    try:
        configuration = ProstateConfiguration.model_validate(value)
    except ValidationError as error:
        raise InvalidWorkflowInvocation(
            "resolved configuration does not match prostate heatmap workflow"
        ) from error
    if configuration.model_fields_set != ProstateConfiguration.model_fields.keys():
        raise InvalidWorkflowInvocation(
            "resolved configuration does not match prostate heatmap workflow"
        )
    return configuration
