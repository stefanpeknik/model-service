"""What callers and providers see: requests, views, admission and outcome types, errors.

Lease fields and protected provider context live in internal_state.py and never appear
in an ExecutionView; statuses and error codes are in status.py.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from model_service.contracts.failures import (
    IncompatibleModelError,
    UnknownBuilderError,
    UnknownModelError,
)
from model_service.contracts.invocation import Invocation
from model_service.contracts.json_values import JsonValue
from model_service.contracts.model_declarations import ModelBinding
from model_service.contracts.result import BuilderResult
from model_service.scheduler.status import ExecutionStatus


if TYPE_CHECKING:
    from datetime import datetime


__all__ = ["IncompatibleModelError", "UnknownBuilderError", "UnknownModelError"]


# The integration a scheduled execution belongs to; each provider names itself.
type ProviderName = str


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    """`body` is provider-specific; standalone bodies are a `BuilderRequest`."""

    builder_id: str
    model_id: str
    provider: ProviderName
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


type ModelResolver = Callable[[str, str], ModelBinding]
"""Return the current model binding for (builder_id, model_id), raising
UnknownBuilderError, UnknownModelError or IncompatibleModelError."""


@dataclass(frozen=True, slots=True)
class ExecutionError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ExecutionView:
    id: str
    builder_id: str
    model_id: str
    status: ExecutionStatus
    result: BuilderResult | None = None
    error: ExecutionError | None = None
    # A failure or cancellation not yet reported to the provider.
    provider_pending: bool = False


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    execution: ExecutionView
    created: bool


class InvalidScheduleRequestError(ValueError):
    pass


class ScheduleConflictError(RuntimeError):
    """The submission identity was reused for a different request."""


class SchedulingUnavailableError(RuntimeError):
    pass


class UnknownExecutionError(KeyError):
    pass


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
