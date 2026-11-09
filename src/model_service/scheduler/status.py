"""The states, kinds and error codes of an execution."""

from __future__ import annotations

from enum import StrEnum


class ExecutionStatus(StrEnum):
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
    REQUEST_UNAVAILABLE = "inference_request_unavailable"
    PROTECTED_CONTEXT_UNREADABLE = "protected_context_unreadable"
