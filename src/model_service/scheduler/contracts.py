"""Schedule requests, execution statuses and safe failure types.

Lease fields and protected provider context live in domain.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from model_service.builders.contracts import JsonValue


class ExecutionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionErrorCode(StrEnum):
    PROVIDER_FINISH_OUTCOME_UNKNOWN = "provider_finish_outcome_unknown"
    PROCESSING_ATTEMPTS_EXHAUSTED = "processing_attempts_exhausted"
    PROVIDER_FINISH_ATTEMPTS_EXHAUSTED = "provider_finish_attempts_exhausted"
    DEADLINE_EXCEEDED = "execution_deadline_exceeded"


# The integration a scheduled execution belongs to; each provider names itself.
type ProviderName = str


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    """`body` is provider-specific."""

    builder_id: str
    model_id: str
    provider: ProviderName
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionError:
    code: str
    message: str


class ScheduleConflictError(RuntimeError):
    """The submission identity was reused for a different request."""


class UnknownExecutionError(KeyError):
    pass
