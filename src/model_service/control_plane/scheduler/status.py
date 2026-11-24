"""The statuses and error codes of an execution."""

from __future__ import annotations

from enum import StrEnum


class ExecutionStatus(StrEnum):
    """Where an execution is in its lifecycle."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """Whether an execution in this status is finished for good."""
        return self in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }


class ExecutionErrorCode(StrEnum):
    """Why the scheduler failed an execution, beside the runner's failures."""

    RUN_OUTCOME_UNKNOWN = "run_outcome_unknown"
    PROVIDER_FINISH_OUTCOME_UNKNOWN = "provider_finish_outcome_unknown"
    PROCESSING_ATTEMPTS_EXHAUSTED = "processing_attempts_exhausted"
    PROVIDER_FINISH_ATTEMPTS_EXHAUSTED = "provider_finish_attempts_exhausted"
    DEADLINE_EXCEEDED = "execution_deadline_exceeded"
    PROTECTED_CONTEXT_UNREADABLE = "protected_context_unreadable"
