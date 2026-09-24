from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime

    from execution_scheduler.contracts import ExecutionError, ExecutionStatus
    from workflow_execution.contracts import (
        JsonValue,
        WorkflowInvocation,
        WorkflowResult,
    )


class WorkKind(StrEnum):
    PROCESSING = "processing"
    PROVIDER_FINISH = "provider_finish"
    PROVIDER_FAILURE_REPORT = "provider_failure_report"
    PROVIDER_CANCELLATION_REPORT = "provider_cancellation_report"


class RenewDirective(StrEnum):
    CONTINUE = "continue"
    CANCEL = "cancel"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    LEASE_LOST = "lease_lost"


class StaleWorkClaimError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    initial_backoff: timedelta
    maximum_backoff: timedelta

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.initial_backoff < timedelta(0):
            raise ValueError("initial_backoff cannot be negative")
        if self.maximum_backoff < self.initial_backoff:
            raise ValueError("maximum_backoff cannot be shorter than initial_backoff")

    def backoff(self, attempt: int) -> timedelta:
        multiplier = 2 ** max(0, attempt - 1)
        return min(self.initial_backoff * multiplier, self.maximum_backoff)


@dataclass(frozen=True, slots=True)
class SubmissionIdentity:
    provider: str
    key: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class PreparedWorkflow:
    invocation: WorkflowInvocation
    protected_context: dict[str, JsonValue]
    deadline_cap: datetime | None = None
    terminal_report_deadline_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Admission:
    identity: SubmissionIdentity
    workflow_id: str
    invocation: WorkflowInvocation
    protected_context: dict[str, JsonValue]
    deadline_at: datetime
    terminal_report_deadline_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    id: str
    identity: SubmissionIdentity
    workflow_id: str
    status: ExecutionStatus
    invocation: WorkflowInvocation
    protected_context: dict[str, JsonValue]
    deadline_at: datetime
    terminal_report_deadline_at: datetime | None
    work_kind: WorkKind | None
    next_attempt_at: datetime | None
    processing_attempts: int
    processing_rejections: int
    finish_attempts: int
    finish_mutation_started: bool
    lease_owner: str | None
    fencing_token: str | None
    lease_expires_at: datetime | None
    result: WorkflowResult | None
    error: ExecutionError | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    execution: ExecutionRecord
    created: bool


@dataclass(frozen=True, slots=True)
class WorkClaim:
    execution_id: str
    workflow_id: str
    provider: str
    kind: WorkKind
    invocation: WorkflowInvocation
    protected_context: dict[str, JsonValue]
    deadline_at: datetime
    attempt: int
    fencing_token: str
    lease_expires_at: datetime
    result: WorkflowResult | None
    error: ExecutionError | None


@dataclass(frozen=True, slots=True)
class ProcessingSucceeded:
    result: WorkflowResult


@dataclass(frozen=True, slots=True)
class ProviderFinishSucceeded:
    pass


@dataclass(frozen=True, slots=True)
class ProcessingRejected:
    retry_after: timedelta | None = None


@dataclass(frozen=True, slots=True)
class ProcessingFailed:
    error: ExecutionError
    retryable: bool


@dataclass(frozen=True, slots=True)
class ProviderFinishFailed:
    error: ExecutionError
    retryable: bool
    outcome_unknown: bool = False


type Settlement = (
    ProcessingSucceeded
    | ProcessingRejected
    | ProcessingFailed
    | ProviderFinishSucceeded
    | ProviderFinishFailed
)
