"""The errors every process shares, each safe to show to a caller."""

from __future__ import annotations

from enum import StrEnum


class FailureCode(StrEnum):
    INPUT_UNAVAILABLE = "input_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_FAILED = "model_failed"
    MODEL_CONTRACT_MISMATCH = "model_contract_mismatch"
    INVALID_MODEL_OUTPUT = "invalid_model_output"


class InvalidInvocation(ValueError):
    """Inputs, access, configuration or deadline violate the builder's contract."""


class UnknownBuilderError(KeyError):
    pass


class IncompatibleModelError(InvalidInvocation):
    """The model's inference contract is not the one the builder accepts."""


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
