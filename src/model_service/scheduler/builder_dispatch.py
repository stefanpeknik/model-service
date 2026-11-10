"""How the scheduler sends a builder run to Ray Serve, and how a send can end."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import timedelta

    from model_service.inference.model_declarations import ModelBinding
    from model_service.shared.invocation import ExecutionContext, Invocation
    from model_service.shared.result import BuilderResult


class BuilderDispatch(ABC):
    @abstractmethod
    async def run(
        self,
        builder_id: str,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext,
    ) -> BuilderResult:
        """Run a builder for one invocation attempt.

        Args:
            builder_id: Registered builder id.
            invocation: Invocation to run.
            model: Model binding resolved for this attempt.
            execution_context: Attempt identity and protected provider state.

        Returns:
            The builder's result.
        """
        ...


class ConfirmedRejectionError(RuntimeError):
    def __init__(self, retry_after: timedelta | None = None) -> None:
        """Record a rejection that happened before the builder started work.

        Args:
            retry_after: Delay the builder asked for, if it sent one.
        """
        super().__init__("builder rejected the request before starting it")
        self.retry_after = retry_after


class UnknownOutcomeError(RuntimeError):
    pass


class RemoteFailureError(RuntimeError):
    def __init__(self, *, code: str, safe_message: str, retryable: bool) -> None:
        """Record a safe failure the builder reported.

        Args:
            code: Stable failure code from the builder.
            safe_message: Message safe to show callers.
            retryable: Whether the builder considers a retry worthwhile.
        """
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
