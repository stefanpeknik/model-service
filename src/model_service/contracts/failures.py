"""The errors every process shares, each safe to show to a caller."""

from __future__ import annotations

from enum import StrEnum


class FailureCode(StrEnum):
    """Why a run failed."""

    DEADLINE_EXCEEDED = "run_deadline_exceeded"
    INTERNAL_ERROR = "run_internal_error"
    INPUT_UNAVAILABLE = "input_unavailable"
    MODEL_FAILED = "model_failed"
    MODEL_CONTRACT_MISMATCH = "model_contract_mismatch"
    INVALID_MODEL_OUTPUT = "invalid_model_output"


def failure_status(retryable: bool) -> int:
    """The HTTP status a caller-safe run failure is answered with.

    Args:
        retryable: Whether a retry may succeed.

    Returns:
        503 for a retryable failure, 422 for one that will fail again.
    """
    return 503 if retryable else 422


class InvalidInvocation(ValueError):
    """Inputs, access, settings or deadline don't fit what is run."""


class UnknownModelError(KeyError):
    """No model has the requested id."""


class UnknownPipelineError(KeyError):
    """No deployed pipeline has the requested id."""


class RunFailure(RuntimeError):
    """A caller-safe failure with a stable code and a retry decision."""

    def __init__(self, *, code: str, safe_message: str, retryable: bool) -> None:
        """Record a caller-safe failure.

        Args:
            code: Stable failure code, usually a `FailureCode`.
            safe_message: Message safe to show callers.
            retryable: Whether a retry may succeed.
        """
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
