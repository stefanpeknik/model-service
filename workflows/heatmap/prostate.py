"""Implement the provider-neutral prostate heatmap workflow and its configuration.

The processor writes a temporary TIFF. Atomic replacement publishes the completed
artifact, preventing results from referencing partially written output files.
"""

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
    """Define the file-producing computation boundary injected into the prostate workflow."""

    async def build(self, request: HeatmapBuildRequest) -> None:
        """Build the requested TIFF at its destination, propagating errors or cancellation.

        Args:
            request: Slide/mask files, destination, stride fraction, and TIFF tile
                geometry.

        Returns:
            None: No return value.
        """
        ...


class ProstateHeatmapWorkflow:
    """Convert declared slide and region inputs into one atomically published heatmap artifact."""

    def __init__(self, processor: HeatmapProcessor) -> None:
        """Retain the processor used to perform slide computation.

        Args:
            processor: Injected file-producing heatmap computation service.
        """
        self._processor = processor

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """Validate resolved configuration, build a unique temporary TIFF, atomically replace the final path, and return its artifact reference. Remove the temporary file on every exit.

        Args:
            context: Invocation-scoped input/output access, resolved configuration, and
                absolute deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.

        Raises:
            InvalidWorkflowInvocation: Configuration or input/output access violates the
                workflow contract.
            OSError: Creating, replacing, or removing an output file fails.
            RuntimeError: The processor does not produce a regular TIFF file, or the
                completed destination is not a regular file.
            asyncio.CancelledError: The caller cancels processing; the temporary output
                is removed before propagation.
            Exception: The injected processor fails; its original exception propagates
                after temporary-file cleanup.
        """
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
    """Validate the relative output path, tile stride fraction, and positive TIFF tile dimensions.

    Attributes:
        output_path: Destination artifact or configured relative output path.
        stride_fraction: Tile step as a positive fraction of model input extent, at most
            one.
        output_tile_height: TIFF storage tile height in pixels.
        output_tile_width: TIFF storage tile width in pixels.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    output_path: str = Field(default="heatmap.tiff", min_length=1)
    stride_fraction: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    output_tile_height: int = Field(default=256, gt=0)
    output_tile_width: int = Field(default=256, gt=0)

    def as_values(self) -> dict[str, JsonValue]:
        """Return JSON-compatible configuration values for defaults and invocation preparation.

        Returns:
            dict[str, JsonValue]: Complete output path, stride fraction, and TIFF tile
                settings.
        """
        return self.model_dump()


def default_configuration() -> dict[str, JsonValue]:
    """Return the complete built-in prostate workflow configuration.

    Returns:
        dict[str, JsonValue]: Complete built-in output path, stride, and TIFF tile
            defaults.
    """
    return ProstateConfiguration().as_values()


def validate_configuration(value: Mapping[str, object]) -> ProstateConfiguration:
    """Require a valid fully resolved configuration with every field explicitly present; translate validation failure into InvalidWorkflowInvocation.

    Args:
        value: Resolved prostate settings; every field must be explicitly supplied and
            valid.

    Returns:
        ProstateConfiguration: Fully resolved validated heatmap configuration.

    Raises:
        InvalidWorkflowInvocation: Inputs, resolved configuration, access context, or
            deadline violate the workflow contract.
    """
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
