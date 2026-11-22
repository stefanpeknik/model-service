"""The errors every process shares, each safe to show to a caller."""

from __future__ import annotations

from enum import StrEnum


class FailureCode(StrEnum):
    """Why a run failed; the `builder_` codes keep their names for callers."""

    DEADLINE_EXCEEDED = "builder_deadline_exceeded"
    INTERNAL_ERROR = "builder_internal_error"
    INPUT_UNAVAILABLE = "input_unavailable"
    MODEL_FAILED = "model_failed"
    MODEL_CONTRACT_MISMATCH = "model_contract_mismatch"
    INVALID_MODEL_OUTPUT = "invalid_model_output"


class InvalidInvocation(ValueError):
    """Inputs, access, settings or deadline don't fit what is run."""


class UnknownModelError(KeyError):
    """No model has the requested id."""


class UnknownAnalysisError(KeyError):
    """No deployed analysis has the requested id."""


class BuilderFailure(RuntimeError):
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
