"""Define internal execution records, leases, retry policies, and worker outcomes.

These values carry state between orchestration, transitions, and persistence. They do
not themselves write to storage or start work.
"""

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
    """Distinguish workflow computation from result publication and terminal status reporting."""

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
    """Bound attempts and exponential delays separately for processing and provider work.

    Attributes:
        max_attempts: Maximum consumed attempts permitted by this policy.
        initial_backoff: Delay after the first retryable outcome.
        maximum_backoff: Upper bound on exponential retry delay.
    """

    max_attempts: int
    initial_backoff: timedelta
    maximum_backoff: timedelta

    def __post_init__(self) -> None:
        """Reject nonpositive attempt limits, negative delays, and a maximum below the initial delay.

        Raises:
            ValueError: max_attempts is nonpositive, initial_backoff is negative, or
                maximum_backoff is shorter than initial_backoff.
        """
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.initial_backoff < timedelta(0):
            raise ValueError("initial_backoff cannot be negative")
        if self.maximum_backoff < self.initial_backoff:
            raise ValueError("maximum_backoff cannot be shorter than initial_backoff")

    def backoff(self, attempt: int) -> timedelta:
        """Return the delay for a one-based attempt, doubling until the configured maximum without overflow.

        Args:
            attempt: One-based retry attempt number, or the fixture attempt identity
                when constructing access.

        Returns:
            timedelta: Delay for the one-based attempt, exponentially doubled and capped
                at maximum_backoff.
        """
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
    """Scope a replay key to its provider and distinguish identical requests from conflicting ones.

    Attributes:
        provider: Integration name selecting identity, preparation, and terminal
            reporting behavior.
        key: Provider-scoped submission replay key.
        fingerprint: Canonical request digest used to reject conflicting replay
            contents.
    """

    provider: str
    key: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class PreparedWorkflow:
    """Carry a validated invocation, private provider state, and any provider-imposed deadline limits.

    Attributes:
        invocation: Prepared typed workflow inputs, configuration, access state, and
            absolute deadline.
        protected_context: Private provider JSON, encrypted in PostgreSQL and excluded
            from public views.
        deadline_cap: Optional provider-imposed upper bound on execution lifetime.
        terminal_report_deadline_at: Optional later limit for reporting
            failure/cancellation to the provider.
    """

    invocation: WorkflowInvocation
    protected_context: dict[str, JsonValue]
    deadline_cap: datetime | None = None
    terminal_report_deadline_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Admission:
    """Carry the complete prepared submission that persistence must admit atomically.

    Attributes:
        identity: Provider-scoped replay key and request fingerprint.
        workflow_id: Declared workflow identifier selecting configuration and dispatch
            routing.
        invocation: Prepared typed workflow inputs, configuration, access state, and
            absolute deadline.
        protected_context: Private provider JSON, encrypted in PostgreSQL and excluded
            from public views.
        deadline_at: Absolute aware deadline, including input acquisition and execution.
        terminal_report_deadline_at: Optional later limit for reporting
            failure/cancellation to the provider.
    """

    identity: SubmissionIdentity
    workflow_id: str
    invocation: WorkflowInvocation
    protected_context: dict[str, JsonValue]
    deadline_at: datetime
    terminal_report_deadline_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Snapshot all durable state for one execution, including pending work, leases, and stored outcomes.

    Processing may have a stored result while status remains running, because provider
    finish is still outstanding. Protected context can contain credentials and must not
    enter public views.

    Attributes:
        id: Durable scheduler execution identifier.
        identity: Provider-scoped replay key and request fingerprint.
        workflow_id: Declared workflow identifier selecting configuration and dispatch
            routing.
        status: Public queued, running, succeeded, failed, or cancelled execution
            status.
        invocation: Prepared typed workflow inputs, configuration, access state, and
            absolute deadline.
        protected_context: Private provider JSON, encrypted in PostgreSQL and excluded
            from public views.
        deadline_at: Absolute aware deadline, including input acquisition and execution.
        terminal_report_deadline_at: Optional later limit for reporting
            failure/cancellation to the provider.
        work_kind: Pending computation/publication/report phase, or None when no work
            remains.
        next_attempt_at: Earliest eligible retry time, or None when no work remains.
        processing_attempts: Number of computation attempts consumed; confirmed
            preacceptance rejections restore this budget.
        processing_rejections: Number of confirmed preacceptance rejections used for
            backoff.
        finish_attempts: Number of provider completion/report attempts consumed.
        finish_mutation_started: Whether an external provider write began, making
            interrupted outcome recovery uncertain.
        lease_owner: Worker owning the current claim, or None without a lease.
        fencing_token: Current attempt token required for durable mutations, or None
            without a lease.
        lease_expires_at: Time ownership ends without renewal, or None without a lease.
        result: Stored workflow outputs; public visibility waits for successful provider
            finish.
        error: Safe terminal error, if established.
        created_at: Initial durable admission timestamp.
        updated_at: Timestamp of the latest durable state change.
    """

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
    """Pair the admitted record with a flag distinguishing insertion from an idempotent replay.

    Attributes:
        execution: Immutable execution record after this state decision.
        created: True for insertion and False for an identical replay.
    """

    execution: ExecutionRecord
    created: bool


@dataclass(frozen=True, slots=True)
class WorkClaim:
    """Give a worker the inputs and token for one leased processing or provider attempt.

    Attributes:
        execution_id: Identifier of the durable execution containing this attempt.
        workflow_id: Workflow identifier selecting configuration and dispatch routing.
        provider: Integration name selecting preparation and terminal reporting
            behavior.
        kind: Processing, publication, failure-report, or cancellation-report phase
            claimed by the worker.
        invocation: Prepared workflow inputs, resolved configuration, access state, and
            absolute deadline.
        protected_context: Private provider JSON; PostgreSQL encrypts it at rest.
        attempt: One-based attempt counter for this processing or provider phase.
        fencing_token: Unique token required for renewal, settlement, and provider
            mutation.
        result: Stored computation outputs, if publication is due.
        error: Stored safe terminal failure, if failure reporting is due.
    """

    execution_id: str
    workflow_id: str
    provider: str
    kind: WorkKind
    invocation: WorkflowInvocation
    protected_context: dict[str, JsonValue]
    attempt: int
    fencing_token: str
    result: WorkflowResult | None
    error: ExecutionError | None


@dataclass(frozen=True, slots=True)
class ProcessingSucceeded:
    """Report a computation result to store before scheduling provider finish.

    Attributes:
        result: Stored workflow outputs; public visibility waits for successful provider
            finish.
    """

    result: WorkflowResult


@dataclass(frozen=True, slots=True)
class ProviderFinishSucceeded:
    """Report that publication or terminal reporting completed successfully."""


@dataclass(frozen=True, slots=True)
class ProcessingRejected:
    """Report confirmed rejection before acceptance, optionally with a minimum retry delay.

    Attributes:
        retry_after: Optional minimum delay supplied by confirmed preacceptance
            rejection.
    """

    retry_after: timedelta | None = None


@dataclass(frozen=True, slots=True)
class ProcessingFailed:
    """Report a safe processing error and whether another attempt is permitted.

    Attributes:
        error: Safe terminal error, if established.
        retryable: Whether this classified failure permits another attempt.
    """

    error: ExecutionError
    retryable: bool


@dataclass(frozen=True, slots=True)
class ProviderFinishFailed:
    """Report a provider failure, including whether a remote mutation's outcome is uncertain.

    Attributes:
        error: Safe terminal error, if established.
        retryable: Whether this classified failure permits another attempt.
        outcome_unknown: Whether an external mutation may have happened without
            confirmation.
    """

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
