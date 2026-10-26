"""Public scheduler requests, execution views and safe failure types.

Lease fields and protected provider context live in domain.py and never appear
in an ExecutionView.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from pydantic import Field, TypeAdapter

from model_service.builders.contracts import (
    BuilderResult,
    Deadline,
    IncompatibleModelError,
    Invocation,
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


class Operation(StrEnum):
    """What an execution runs: a builder over a slide, or one raw model request."""

    BUILDER = "builder"
    INFERENCE = "inference"


class ExecutionErrorCode(StrEnum):
    BUILDER_OUTCOME_UNKNOWN = "builder_outcome_unknown"
    PROVIDER_FINISH_OUTCOME_UNKNOWN = "provider_finish_outcome_unknown"
    PROCESSING_ATTEMPTS_EXHAUSTED = "processing_attempts_exhausted"
    PROVIDER_FINISH_ATTEMPTS_EXHAUSTED = "provider_finish_attempts_exhausted"
    DEADLINE_EXCEEDED = "execution_deadline_exceeded"
    UNKNOWN_MODEL = "unknown_model"
    INCOMPATIBLE_MODEL = "incompatible_model"
    INCOMPATIBLE_INPUT = "incompatible_input"
    MODEL_REJECTED_INPUT = "model_rejected_input"
    MODEL_FAILED = "model_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    RESULT_TOO_LARGE = "inference_result_too_large"


class ProviderName(StrEnum):
    """The integration a scheduled execution belongs to."""

    STANDALONE = "standalone"
    EMPAIA = "empaia"
    # Scheduled raw inference; it has no provider adapter.
    INFERENCE = "inference"


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    """`body` is provider-specific; standalone bodies are a `BuilderRequest`."""

    builder_id: str
    model_id: str
    provider: ProviderName
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceScheduleRequest:
    """One raw model request, exactly as `POST /{model_id}/` would receive it."""

    model_id: str
    payload: bytes
    headers: dict[str, str]
    idempotency_key: str
    requested_timeout_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class InferenceInvocation:
    """The stored part of a raw request; the payload bytes are kept separately."""

    headers: Annotated[dict[str, str], Field(strict=True)]
    deadline_at: Deadline


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Describes a stored model response; its bytes are served by `/result`."""

    media_type: str | None
    headers: Annotated[dict[str, str], Field(strict=True)]
    size: Annotated[int, Field(strict=True, ge=0)]


type StoredInvocation = Invocation | InferenceInvocation
type StoredResult = BuilderResult | InferenceResult
STORED_INVOCATION: TypeAdapter[StoredInvocation] = TypeAdapter(StoredInvocation)
STORED_RESULT: TypeAdapter[StoredResult] = TypeAdapter(StoredResult)


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
    operation: Operation
    builder_id: str | None
    model_id: str
    status: ExecutionStatus
    result: StoredResult | None = None
    error: ExecutionError | None = None
    # A failure or cancellation not yet reported to the provider (e.g. EMPAIA).
    provider_pending: bool = False
    # Why a running execution makes no progress, when known.
    message: str | None = None


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


class ResultUnavailableError(RuntimeError):
    """The execution has no stored inference response to download."""
