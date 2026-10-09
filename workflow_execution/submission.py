"""Turn public workflow requests into internal invocations without processing settings.

Public callers supply data access and an optional artifact destination, never
processing configuration. The workflow worker chooses its active defaults at execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

from workflow_execution.contracts import (
    InputAccessContext,
    WorkflowDeadline,
    WorkflowInput,
    WorkflowInvocation,
)


if TYPE_CHECKING:
    from datetime import datetime


class WorkflowRequest(BaseModel):
    """Accept public input references and an optional per-execution destination.

    Attributes:
        inputs: Named data references accepted by the selected workflow.
        input_access: Public data-access adapter and its non-secret settings.
        output_path: Optional artifact path beneath the permitted output root.
            None uses the workflow's server-owned destination default.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: Annotated[dict[str, WorkflowInput], Field(strict=True)]
    input_access: InputAccessContext
    output_path: Annotated[str, Field(strict=True, min_length=1)] | None = None


class ImmediateWorkflowRequest(WorkflowRequest):
    """Include the caller's absolute deadline for immediate execution.

    Attributes:
        deadline_at: Aware absolute deadline covering data access and processing.
    """

    deadline_at: WorkflowDeadline


def prepare_invocation(
    request: WorkflowRequest,
    *,
    deadline_at: datetime,
) -> WorkflowInvocation:
    """Keep data access and destination while deferring server settings to execution.

    Args:
        request: Validated public data references and optional artifact destination.
        deadline_at: Absolute deadline supplied by immediate execution or admission.

    Returns:
        WorkflowInvocation: Standalone invocation with no captured processing settings.
    """
    return WorkflowInvocation(
        inputs=request.inputs,
        resolved_configuration=None,
        input_access=request.input_access,
        deadline_at=deadline_at,
        output_path=request.output_path,
    )
