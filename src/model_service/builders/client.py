"""Call Ray builder replicas over HTTP and classify the outcome.

A 429 is a confirmed rejection before work started. Transport failures and
malformed responses leave the outcome unknown, because the builder may have run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx
from pydantic import TypeAdapter, ValidationError

from model_service.builders.contracts import RESULT
from model_service.builders.http import RUN_REQUEST, ErrorResponse, RunRequest


if TYPE_CHECKING:
    from model_service.builders.contracts import (
        BuilderResult,
        ExecutionContext,
        Invocation,
        ModelBinding,
    )


_ERROR_RESPONSE = TypeAdapter(ErrorResponse)


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


class HTTPBuilderDispatch(BuilderDispatch):
    def __init__(self, client: httpx.AsyncClient, *, origin: str) -> None:
        """Dispatch builder runs to a Ray Serve origin.

        Args:
            client: HTTP client used for the requests.
            origin: Base URL of the Ray Serve HTTP proxy; a trailing slash is dropped.
        """
        self._client = client
        self._origin = origin.rstrip("/")

    async def run(
        self,
        builder_id: str,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext,
    ) -> BuilderResult:
        """Post one run request to a builder replica and classify the outcome.

        The request timeout is the time left until the invocation's deadline.

        Args:
            builder_id: Registered builder id, used in the route.
            invocation: Invocation to run.
            model: Model binding resolved for this attempt.
            execution_context: Attempt identity and protected provider state.

        Returns:
            The builder's result from a 200 response.

        Raises:
            ConfirmedRejectionError: The builder answered 429, or Ray Serve answered a
                non-JSON 503, so the work never started.
            UnknownOutcomeError: The transport failed or the response was malformed,
                so the builder may have run.
            RemoteFailureError: The builder returned a well-formed error envelope.
        """
        timeout = max(
            (invocation.deadline_at - datetime.now(UTC)).total_seconds(), 0.001
        )
        request = RunRequest(invocation, model, execution_context)
        try:
            response = await self._client.post(
                f"{self._origin}/builders/{quote(builder_id, safe='')}/",
                json=RUN_REQUEST.dump_python(request, mode="json"),
                timeout=timeout,
            )
        except httpx.TransportError as error:
            raise UnknownOutcomeError("builder dispatch transport failed") from error
        if response.status_code == 429:
            raise ConfirmedRejectionError(
                _retry_after(response.headers.get("retry-after"))
            )
        try:
            body = response.json()
        except ValueError as error:
            # Ray Serve's own 503 (replica queue full, none running): our builder
            # answers JSON, so the request never reached it.
            if response.status_code == 503:
                raise ConfirmedRejectionError(
                    _retry_after(response.headers.get("retry-after"))
                ) from error
            raise UnknownOutcomeError("builder returned an invalid response") from error
        if response.status_code == 200:
            try:
                return RESULT.validate_python(body)
            except ValidationError as error:
                raise UnknownOutcomeError(
                    "builder returned an invalid result"
                ) from error
        try:
            detail = _ERROR_RESPONSE.validate_python(body).detail
        except ValidationError as error:
            raise UnknownOutcomeError(
                f"builder returned HTTP {response.status_code}"
            ) from error
        raise RemoteFailureError(
            code=detail.code, safe_message=detail.message, retryable=detail.retryable
        )


def _retry_after(value: str | None) -> timedelta | None:
    """Parse a `Retry-After` header given in seconds.

    Args:
        value: Header value, or None when absent.

    Returns:
        The delay, or None when the header is absent, negative or not an integer
        (HTTP dates are not supported).
    """
    try:
        seconds = int(value) if value is not None else -1
    except ValueError:
        return None
    return timedelta(seconds=seconds) if seconds >= 0 else None
