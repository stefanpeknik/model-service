"""Define the public scheduler requests, execution views, and safe failure types.

Internal lease fields and protected provider context live in domain.py and never appear
in ExecutionView.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from workflow_execution.contracts import JsonValue, WorkflowResult  # noqa: TC001


class ExecutionStatus(StrEnum):
    """Describe caller-visible progress, including queued work and terminal outcomes."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionErrorCode(StrEnum):
    """Name scheduler failures independently of workflow and provider error codes."""

    WORKFLOW_OUTCOME_UNKNOWN = "workflow_outcome_unknown"
    PROVIDER_FINISH_OUTCOME_UNKNOWN = "provider_finish_outcome_unknown"
    PROCESSING_ATTEMPTS_EXHAUSTED = "processing_attempts_exhausted"
    PROVIDER_FINISH_ATTEMPTS_EXHAUSTED = "provider_finish_attempts_exhausted"
    DEADLINE_EXCEEDED = "execution_deadline_exceeded"


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    """Carry a provider-specific body, replay key, and optional end-to-end execution timeout.

    Attributes:
        workflow_id: Declared workflow identifier selecting configuration and dispatch
            routing.
        provider: Integration name selecting identity, preparation, and terminal
            reporting behavior.
        body: Provider-specific JSON payload submitted for preparation.
        requested_timeout_seconds: Optional positive caller execution budget in seconds.
        idempotency_key: Caller replay key; required for standalone admission.
    """

    workflow_id: str
    provider: str
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionError:
    """Carry an error code and message safe to expose through the public API.

    Attributes:
        code: Stable safe failure identifier.
        message: Safe user-facing failure explanation.
    """

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ExecutionView:
    """Expose execution status, with results only on success and errors only on failure.

    Attributes:
        id: Durable scheduler execution identifier.
        workflow_id: Declared workflow identifier selecting configuration and dispatch
            routing.
        status: Public queued, running, succeeded, failed, or cancelled execution
            status.
        result: Stored workflow outputs; public visibility waits for successful provider
            finish.
        error: Safe terminal error, if established.
    """

    id: str
    workflow_id: str
    status: ExecutionStatus
    result: WorkflowResult | None = None
    error: ExecutionError | None = None


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    """Return the execution view and whether admission created a new record or replayed one.

    Attributes:
        execution: Public execution view for the newly admitted or replayed submission.
        created: True for insertion and False for an identical replay.
    """

    execution: ExecutionView
    created: bool


class UnknownWorkflowError(ValueError):
    """Signal a workflow ID absent from the scheduler's configured catalog."""


class InvalidScheduleRequestError(ValueError):
    """Signal a request that cannot be prepared or admitted as scheduled work."""


class ScheduleConflictError(RuntimeError):
    """Signal reuse of a provider's submission key with a different request fingerprint."""


class SchedulingUnavailableError(RuntimeError):
    """Signal storage unavailability that the public API maps to a retryable service error."""


class UnknownExecutionError(KeyError):
    """Signal a read or cancellation request for an execution ID that does not exist."""
