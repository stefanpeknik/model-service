"""Define the internal workflow HTTP request and safe error response contract.

The request combines the public invocation with separate protected execution context; it
is sent only across the internal dispatch boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from fastapi import HTTPException, status
from pydantic import TypeAdapter

from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    WorkflowContractViolation,
    WorkflowExecutionContext,
    WorkflowFailure,
    WorkflowInvocation,
)


if TYPE_CHECKING:
    from workflow_execution.contracts import WorkflowResult


@dataclass(frozen=True, slots=True)
class WorkflowRunRequest(WorkflowInvocation):
    """Extend an invocation with private execution context for internal workflow dispatch.

    Attributes:
        execution_context: Private processing identity and protected provider state in
            internal dispatch.
    """

    execution_context: WorkflowExecutionContext = field(
        default_factory=WorkflowExecutionContext
    )

    @classmethod
    def from_invocation(
        cls,
        invocation: WorkflowInvocation,
        execution_context: WorkflowExecutionContext,
    ) -> WorkflowRunRequest:
        """Combine invocation fields and execution context into the internal request envelope.

        Args:
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.
            execution_context: Private attempt identity and provider context kept
                separate from the invocation.

        Returns:
            WorkflowRunRequest: Internal HTTP envelope combining invocation and
                protected execution context.
        """
        return cls(
            inputs=invocation.inputs,
            resolved_configuration=invocation.resolved_configuration,
            input_access=invocation.input_access,
            deadline_at=invocation.deadline_at,
            execution_context=execution_context,
        )


_RUN_REQUEST = TypeAdapter(WorkflowRunRequest)


def run_request_to_dict(request: WorkflowRunRequest) -> dict[str, object]:
    """Serialize the internal request, including attempt and protected context, for HTTP transport.

    Args:
        request: Internal invocation envelope containing separate protected execution
            context.

    Returns:
        dict[str, object]: Internal HTTP JSON envelope including the separate private
            execution context.
    """
    return _RUN_REQUEST.dump_python(request, mode="json")


class WorkflowRunner(Protocol):
    """Define the application run method required by the HTTP error adapter."""

    async def run(
        self,
        invocation: WorkflowInvocation,
        execution_context: WorkflowExecutionContext | None = None,
    ) -> WorkflowResult:
        """Execute a typed invocation with optional private attempt context and return typed outputs.

        Args:
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.
            execution_context: Private attempt identity and provider context kept
                separate from the invocation.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.

        Raises:
            InvalidWorkflowInvocation: Inputs, resolved configuration, access context,
                or deadline violate the workflow contract.
            WorkflowContractViolation: The computation returned invalid or incomplete
                declared outputs.
            WorkflowFailure: Execution or resource cleanup fails with a safe classified
                workflow error.
            asyncio.CancelledError: The owning task, caller, deadline, or lease monitor
                cancels the operation.
        """
        ...


class WorkflowHTTPErrorCode(StrEnum):
    """Name HTTP boundary failures for invalid requests, unavailable deployments, and broken output contracts."""

    INVALID_INVOCATION = "invalid_workflow_invocation"
    UNAVAILABLE = "workflow_unavailable"
    CONTRACT_VIOLATION = "workflow_contract_violation"


@dataclass(frozen=True, slots=True)
class WorkflowErrorDetail:
    """Carry a safe code/message and explicit retry decision in workflow HTTP failures.

    Attributes:
        code: Stable safe failure identifier.
        message: Safe user-facing failure explanation.
        retryable: Whether this classified failure permits another attempt.
    """

    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class WorkflowErrorResponse:
    """Wrap safe workflow error details in the standard response envelope.

    Attributes:
        detail: Structured safe HTTP error information.
    """

    detail: WorkflowErrorDetail


async def run_workflow_http(
    application: WorkflowRunner,
    invocation: WorkflowInvocation,
    execution_context: WorkflowExecutionContext | None = None,
) -> WorkflowResult:
    """Run the application and translate known invocation, contract, and workflow failures into structured HTTP errors.

    Args:
        application: Workflow runner whose errors are translated into safe HTTP
            responses.
        invocation: Named inputs, resolved configuration, public access state, and
            absolute workflow deadline.
        execution_context: Private attempt identity and provider context kept separate
            from the invocation.

    Returns:
        WorkflowResult: Named inline outputs and/or references to completed artifact
            files.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
        asyncio.CancelledError: The owning task, caller, deadline, or lease monitor
            cancels the operation.
    """
    try:
        return await application.run(invocation, execution_context)
    except InvalidWorkflowInvocation as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": WorkflowHTTPErrorCode.INVALID_INVOCATION,
                "message": str(error),
                "retryable": False,
            },
        ) from error
    except WorkflowContractViolation as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": WorkflowHTTPErrorCode.CONTRACT_VIOLATION,
                "message": "deployed workflow violated its contract",
                "retryable": False,
            },
        ) from error
    except WorkflowFailure as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
                if error.retryable
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail={
                "code": error.code,
                "message": error.safe_message,
                "retryable": error.retryable,
            },
        ) from error
