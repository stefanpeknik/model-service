"""Public scheduler requests, execution views and safe failure types.

Lease fields and protected provider context live in records.py and never appear
in an ExecutionView; statuses and error codes are in status.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field, TypeAdapter

from model_service.inference.contracts import ModelBinding
from model_service.scheduler.status import (
    ExecutionStatus,  # noqa: TC001 - Pydantic or FastAPI reads it at runtime.
    Operation,  # noqa: TC001 - Pydantic or FastAPI reads it at runtime.
)
from model_service.wire.failures import (
    IncompatibleModelError,
    UnknownBuilderError,
    UnknownModelError,
)
from model_service.wire.invocation import Deadline, Invocation
from model_service.wire.json import (
    JsonValue,  # noqa: TC001 - Pydantic or FastAPI reads it at runtime.
)
from model_service.wire.result import BuilderResult


__all__ = ["IncompatibleModelError", "UnknownBuilderError", "UnknownModelError"]


# The integration a scheduled execution belongs to; each provider names itself.
type ProviderName = str
# Scheduled raw inference; it has no provider adapter.
INFERENCE_PROVIDER: ProviderName = "inference"


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
