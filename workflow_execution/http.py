from dataclasses import dataclass
from enum import StrEnum


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
