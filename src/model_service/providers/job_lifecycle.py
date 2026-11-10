"""What a provider implements for the scheduler: admission and finish of its jobs.

Providers translate caller-specific identity, data access and external job
lifecycle into the scheduler's model. Reading builder inputs belongs to
builders/input_access.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime

    from model_service.scheduler.executions import (
        ExecutionError,
        ProviderName,
        ScheduleRequest,
    )
    from model_service.shared.invocation import Invocation
    from model_service.shared.json_values import JsonValue
    from model_service.shared.result import BuilderResult


type ProviderFinishOutcome = ProviderSuccess | ProviderFailure | ProviderCancellation
type BeforeMutation = Callable[[], Awaitable[None]]


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
class ProviderSuccess:
    """Give provider finish the durable builder result to publish."""

    result: BuilderResult
    # The builder that produced it, whose declarations say what each output is.
    builder_id: str


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """Give provider finish the safe terminal processing error to report."""

    error: ExecutionError


@dataclass(frozen=True, slots=True)
class ProviderCancellation:
    """Ask provider finish to report the execution's cancellation."""


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
            code: Stable error code to record.
            safe_message: Message safe to show to callers.
            retryable: Whether the worker may retry the finish.
            outcome_unknown: Whether a remote mutation may have happened before the failure.
        """
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


class ProviderAdapter(ABC):
    """Translate one integration's replay identity, prepared invocation, and terminal reporting behavior."""

    name: ProviderName

    @abstractmethod
    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Compute submission identity locally without I/O or external side effects.

        Args:
            request: Provider schedule request.

        Returns:
            The replay identity of the request.
        """
        ...

    @abstractmethod
    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedInvocation:
        """Resolve and validate new inputs and configuration before admission.

        Args:
            request: Provider schedule request.
            deadline_at: Execution deadline the preparation must respect.

        Returns:
            The prepared invocation with protected context and deadline limits.
        """
        ...

    @abstractmethod
    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        """Publish or report a stored outcome to the provider.

        Awaits before_mutation before the first external change. Scheduling and
        retries remain worker-owned.

        Args:
            protected_context: Provider context stored at admission.
            outcome: Success, failure or cancellation to report.
            before_mutation: Callback to await before the first external change.

        Raises:
            ProviderFinishFailureError: Publication failed; carries retry permission and whether the outcome is unknown.
        """
        ...
