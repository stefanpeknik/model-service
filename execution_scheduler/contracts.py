from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from workflow_execution.contracts import JsonValue, WorkflowResult  # noqa: TC001


class ExecutionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionErrorCode(StrEnum):
    WORKFLOW_OUTCOME_UNKNOWN = "workflow_outcome_unknown"
    PROCESSING_ATTEMPTS_EXHAUSTED = "processing_attempts_exhausted"
    PROVIDER_FINISH_ATTEMPTS_EXHAUSTED = "provider_finish_attempts_exhausted"
    DEADLINE_EXCEEDED = "execution_deadline_exceeded"


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    workflow_id: str
    provider: str
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ExecutionView:
    id: str
    workflow_id: str
    status: ExecutionStatus
    result: WorkflowResult | None = None
    error: ExecutionError | None = None


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    execution: ExecutionView
    created: bool


class UnknownWorkflowError(ValueError):
    pass


class InvalidScheduleRequestError(ValueError):
    pass


class ScheduleConflictError(RuntimeError):
    pass


class SchedulingUnavailableError(RuntimeError):
    pass


class UnknownExecutionError(KeyError):
    pass
