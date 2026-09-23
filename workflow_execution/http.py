from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from fastapi import HTTPException, status

from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    WorkflowContractViolation,
    WorkflowFailure,
)


if TYPE_CHECKING:
    from workflow_execution.contracts import WorkflowInvocation, WorkflowResult


class WorkflowRunner(Protocol):
    async def run(self, invocation: WorkflowInvocation) -> WorkflowResult: ...


class WorkflowHTTPErrorCode(StrEnum):
    INVALID_INVOCATION = "invalid_workflow_invocation"
    UNAVAILABLE = "workflow_unavailable"
    CONTRACT_VIOLATION = "workflow_contract_violation"


@dataclass(frozen=True, slots=True)
class WorkflowErrorDetail:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class WorkflowErrorResponse:
    detail: WorkflowErrorDetail


async def run_workflow_http(
    application: WorkflowRunner,
    invocation: WorkflowInvocation,
) -> WorkflowResult:
    try:
        return await application.run(invocation)
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
