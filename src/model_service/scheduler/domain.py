"""Define internal execution records, leases, retry policies, and worker outcomes.

These values carry state between orchestration, transitions, and persistence. They do
not themselves write to storage or start work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

from model_service.scheduler.contracts import Operation, ProviderName


if TYPE_CHECKING:
    from datetime import datetime

    from model_service.builders.contracts import (
        BuilderResult,
        Invocation,
        JsonValue,
    )
    from model_service.scheduler.contracts import (
        ExecutionError,
        ExecutionStatus,
        InferenceResult,
        StoredInvocation,
        StoredResult,
    )


class WorkKind(StrEnum):
    """Distinguish builder processing from result publication and terminal status reporting."""

    PROCESSING = "processing"
    PROVIDER_FINISH = "provider_finish"
    PROVIDER_FAILURE_REPORT = "provider_failure_report"
    PROVIDER_CANCELLATION_REPORT = "provider_cancellation_report"


class RenewDirective(StrEnum):
    """Tell a worker to continue or stop because of cancellation, deadline expiry, or lost ownership."""

    CONTINUE = "continue"
    CANCEL = "cancel"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    LEASE_LOST = "lease_lost"


class StaleWorkClaimError(RuntimeError):
    """Reject an operation whose token, lease, deadline, or work kind is no longer valid."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bound attempts and exponential delays separately for processing and provider work."""

    max_attempts: int
    initial_backoff: timedelta
    maximum_backoff: timedelta

    def __post_init__(self) -> None:
        """Reject nonpositive attempt limits, negative delays, and a maximum below the initial delay."""
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.initial_backoff < timedelta(0):
            raise ValueError("initial_backoff cannot be negative")
        if self.maximum_backoff < self.initial_backoff:
            raise ValueError("maximum_backoff cannot be shorter than initial_backoff")

    def backoff(self, attempt: int) -> timedelta:
        """Return the delay for a one-based attempt, doubling until the configured maximum without overflow."""
        delay = self.initial_backoff
        if delay == timedelta(0):
            return delay
        remaining = max(0, attempt - 1)
        while remaining and delay < self.maximum_backoff:
            if delay >= self.maximum_backoff - delay:
                return self.maximum_backoff
            delay *= 2
            remaining -= 1
        return delay


@dataclass(frozen=True, slots=True)
class SubmissionIdentity:
    """Scope a replay key to its provider and distinguish identical requests from conflicting ones."""

    provider: ProviderName
    key: str
    fingerprint: str


class Priority(IntEnum):
    """Order in which due work is claimed; higher first, within each model's capacity.

    Internal for now: providers set it, the public API does not take it.
    """

    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass(frozen=True, slots=True)
class PreparedInvocation:
    """Carry a validated invocation, private provider state, and any provider-imposed deadline limits."""

    invocation: Invocation
    protected_context: dict[str, JsonValue]
    deadline_cap: datetime | None = None
    terminal_report_deadline_at: datetime | None = None
    priority: int = Priority.NORMAL


@dataclass(frozen=True, slots=True)
class Admission:
    """Carry the complete prepared submission that persistence must admit atomically."""

    identity: SubmissionIdentity
    builder_id: str | None
    model_id: str
    invocation: StoredInvocation
    protected_context: dict[str, JsonValue]
    deadline_at: datetime
    terminal_report_deadline_at: datetime | None = None
    operation: Operation = Operation.BUILDER
    payload: bytes | None = None
    priority: int = Priority.NORMAL


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Snapshot all durable state for one execution, including pending work, leases, and stored outcomes."""

    id: str
    identity: SubmissionIdentity
    builder_id: str | None
    model_id: str
    status: ExecutionStatus
    invocation: StoredInvocation
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
    result: StoredResult | None
    error: ExecutionError | None
    created_at: datetime
    updated_at: datetime
    operation: Operation = Operation.BUILDER
    priority: int = Priority.NORMAL

    @property
    def current_attempt(self) -> int:
        """Attempts made at the work now due: processing or provider finish."""
        if self.work_kind is WorkKind.PROCESSING:
            return self.processing_attempts
        return self.finish_attempts


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Pair the admitted record with a flag distinguishing insertion from an idempotent replay."""

    execution: ExecutionRecord
    created: bool


@dataclass(frozen=True, slots=True)
class WorkClaim:
    """Give a worker the inputs and token for one leased processing or provider attempt."""

    execution_id: str
    builder_id: str | None
    model_id: str
    provider: ProviderName
    kind: WorkKind
    invocation: StoredInvocation
    protected_context: dict[str, JsonValue]
    attempt: int
    fencing_token: str
    result: StoredResult | None
    error: ExecutionError | None
    operation: Operation = Operation.BUILDER
    payload: bytes | None = None


@dataclass(frozen=True, slots=True)
class ProcessingSucceeded:
    """Report a computation result to store before scheduling provider finish."""

    result: BuilderResult


@dataclass(frozen=True, slots=True)
class InferenceSucceeded:
    """A raw model response; the repository stores `content` next to the execution."""

    result: InferenceResult
    content: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class ProviderFinishSucceeded:
    """Report that publication or terminal reporting completed successfully."""


@dataclass(frozen=True, slots=True)
class ProcessingRejected:
    """Report confirmed rejection before acceptance, optionally with a minimum retry delay."""

    retry_after: timedelta | None = None


@dataclass(frozen=True, slots=True)
class ProcessingFailed:
    """Report a safe processing error and whether another attempt is permitted."""

    error: ExecutionError
    retryable: bool


@dataclass(frozen=True, slots=True)
class ProviderFinishFailed:
    """Report a provider failure, including whether a remote mutation's outcome is uncertain."""

    error: ExecutionError
    retryable: bool
    outcome_unknown: bool = False


type Settlement = (
    ProcessingSucceeded
    | InferenceSucceeded
    | ProcessingRejected
    | ProcessingFailed
    | ProviderFinishSucceeded
    | ProviderFinishFailed
)
