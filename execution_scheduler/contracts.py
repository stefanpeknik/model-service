"""Public scheduler requests, execution views and safe failure types.

Lease fields and protected provider context live in domain.py and never appear
in an ExecutionView.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from builders.contracts import (
    BuilderResult,
    IncompatibleModelError,
    JsonValue,
    ModelBinding,
    UnknownBuilderError,
    UnknownModelError,
)


__all__ = ["IncompatibleModelError", "UnknownBuilderError", "UnknownModelError"]


class ExecutionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionErrorCode(StrEnum):
    BUILDER_OUTCOME_UNKNOWN = "builder_outcome_unknown"
    PROVIDER_FINISH_OUTCOME_UNKNOWN = "provider_finish_outcome_unknown"
    PROCESSING_ATTEMPTS_EXHAUSTED = "processing_attempts_exhausted"
    PROVIDER_FINISH_ATTEMPTS_EXHAUSTED = "provider_finish_attempts_exhausted"
    DEADLINE_EXCEEDED = "execution_deadline_exceeded"
    UNKNOWN_MODEL = "unknown_model"
    INCOMPATIBLE_MODEL = "incompatible_model"


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    """`body` is provider-specific; standalone bodies are a `BuilderRequest`."""

    builder_id: str
    model_id: str
    provider: str
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
