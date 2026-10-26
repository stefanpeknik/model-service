"""Specify provider preparation and completion around a shared builder execution.

Providers translate caller-specific identity, data access, and external job lifecycle
into the scheduler model. Reading builder inputs belongs to access adapters.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from datetime import datetime

    from model_service.builders.contracts import BuilderResult, JsonValue
    from model_service.scheduler.contracts import (
        ExecutionError,
        ProviderName,
        ScheduleRequest,
    )
    from model_service.scheduler.domain import PreparedInvocation, SubmissionIdentity


@dataclass(frozen=True, slots=True)
class ProviderSuccess:
    """Give provider finish the durable builder result to publish."""

    result: BuilderResult


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """Give provider finish the safe terminal processing error to report."""

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
        """Capture finish failure metadata without exposing the underlying provider exception to callers."""
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


class ProviderAdapter(Protocol):
    """Translate one integration's replay identity, prepared invocation, and terminal reporting behavior."""

    name: ProviderName

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Compute submission identity locally without I/O or external side effects."""
        ...

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedInvocation:
        """Resolve and validate new inputs and configuration before admission, returning protected context and deadline limits."""
        ...

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        """Publish or report a stored outcome, awaiting before_mutation before the first external change. Scheduling and retries remain worker-owned."""
        ...
