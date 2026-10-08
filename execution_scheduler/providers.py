"""Specify provider preparation and completion around a shared workflow execution.

Providers translate caller-specific identity, data access, and external job lifecycle
into the scheduler model. Actual workflow input reading belongs to access adapters.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from datetime import datetime

    from execution_scheduler.contracts import ExecutionError, ScheduleRequest
    from execution_scheduler.domain import PreparedWorkflow, SubmissionIdentity
    from workflow_execution.contracts import JsonValue, WorkflowResult


@dataclass(frozen=True, slots=True)
class ProviderSuccess:
    """Give provider finish the durable workflow result to publish.

    Attributes:
        result: Stored workflow outputs; public visibility waits for successful provider
            finish.
    """

    result: WorkflowResult


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """Give provider finish the safe terminal processing error to report.

    Attributes:
        error: Safe terminal error, if established.
    """

    error: ExecutionError


@dataclass(frozen=True, slots=True)
class ProviderCancellation:
    """Ask provider finish to report the execution's cancellation."""


type ProviderFinishOutcome = ProviderSuccess | ProviderFailure | ProviderCancellation
type BeforeMutation = Callable[[], Awaitable[None]]


class ProviderFinishFailureError(RuntimeError):
    """Carry a safe publication error, retry permission, and whether a remote mutation may have happened."""

    def __init__(
        self,
        *,
        code: str,
        safe_message: str,
        retryable: bool,
        outcome_unknown: bool = False,
    ) -> None:
        """Capture finish failure metadata without exposing the underlying provider exception to callers.

        Args:
            code: Stable failure identifier, or HTTP status when classifying an upstream
                response.
            safe_message: Message permitted in public errors without internal details or
                credentials.
            retryable: Whether another attempt is permitted for this classified failure.
            outcome_unknown: Whether an external mutation may have occurred despite
                missing confirmation.
        """
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


class ProviderAdapter(Protocol):
    """Translate one integration's replay identity, prepared invocation, and terminal reporting behavior.

    Attributes:
        name: Integration identifier used to register and select this adapter.
    """

    name: str

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Compute submission identity locally without I/O or external side effects.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.

        Returns:
            SubmissionIdentity: Provider-scoped replay key and canonical request
                fingerprint.

        Raises:
            InvalidScheduleRequestError: Provider payload, configuration, paths, or
                effective deadline are invalid.
        """
        ...

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedWorkflow:
        """Resolve and validate new inputs and configuration before admission, returning protected context and deadline limits.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.
            deadline_at: Absolute aware deadline for the entire execution, including
                preparation.

        Returns:
            PreparedWorkflow: Validated invocation, private provider state, and
                applicable deadline limits.

        Raises:
            InvalidScheduleRequestError: Provider payload, configuration, paths, or
                effective deadline are invalid.
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
        """
        ...

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        """Publish or report a stored outcome, awaiting before_mutation before the first external change. Scheduling and retries remain worker-owned.

        Args:
            protected_context: Private provider state that persistence encrypts and
                internal dispatch carries separately.
            outcome: Stored success, safe failure, or cancellation to report through the
                provider.
            before_mutation: Async guard awaited before the first external write to
                persist and validate its durable claim.

        Returns:
            None: No return value.

        Raises:
            ProviderFinishFailureError: Publication or terminal reporting fails;
                metadata states whether retry is safe.
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
        """
        ...
