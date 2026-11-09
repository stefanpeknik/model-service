"""Execution statuses and safe failure types.

Lease fields and protected provider context live in domain.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
class ExecutionError:
    code: str
    message: str
