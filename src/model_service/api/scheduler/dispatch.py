"""Send an analysis run to the runner over HTTP, and how a send can end.

A 429 is a confirmed rejection before work started. Transport failures and
malformed responses leave the outcome unknown, because the analysis may have run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
from pydantic import TypeAdapter, ValidationError

from model_service.contracts.analysis_http import ANALYSIS_RUN_REQUEST, ErrorResponse
from model_service.contracts.result import RESULT


if TYPE_CHECKING:
    from model_service.contracts.analysis_http import AnalysisRunRequest
    from model_service.contracts.result import RunResult


_ERROR_RESPONSE = TypeAdapter(ErrorResponse)


class ConfirmedRejectionError(RuntimeError):
    """The runner rejected the request before starting work, so a retry is safe."""

    def __init__(self, retry_after: timedelta | None = None) -> None:
        """Record a rejection that happened before the runner started work.

        Args:
            retry_after: Delay the runner asked for, if it sent one.
        """
        super().__init__("runner rejected the request before starting it")
        self.retry_after = retry_after


class UnknownOutcomeError(RuntimeError):
    """The analysis may or may not have run; the request was lost in flight."""


class RemoteFailureError(RuntimeError):
    """The runner ran the analysis and reported a safe failure."""

    def __init__(self, *, code: str, safe_message: str, retryable: bool) -> None:
        """Record a safe failure the runner reported.

        Args:
            code: Stable failure code from the runner.
            safe_message: Message safe to show callers.
            retryable: Whether the runner considers a retry worthwhile.
        """
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable


class HTTPAnalysisDispatch:
    """Sends analysis runs to the analysis runner over Ray Serve's HTTP proxy."""

    def __init__(self, client: httpx.AsyncClient, *, origin: str) -> None:
        """Dispatch analysis runs to a Ray Serve origin.

        Args:
            client: HTTP client used for the requests.
            origin: Base URL of the Ray Serve HTTP proxy; a trailing slash is dropped.
        """
        self._client = client
        self._origin = origin.rstrip("/")

    async def run_analysis(self, request: AnalysisRunRequest) -> RunResult:
        """Post one resolved analysis to the analysis runner and classify the outcome.

        Args:
            request: The resolved document, the invocation and the attempt.

        Returns:
            The runner's result from a 200 response.

        Raises:
            ConfirmedRejectionError: The runner answered 429, or Ray Serve
                answered a non-JSON 503, so the work never started.
            UnknownOutcomeError: The transport failed or the response was
                malformed, so the analysis may have run.
            RemoteFailureError: The runner returned a well-formed error envelope.
        """
        return await self._post(
            "/analyses/",
            ANALYSIS_RUN_REQUEST.dump_python(request, mode="json"),
            request.invocation.deadline_at,
        )

    async def _post(
        self, path: str, payload: object, deadline_at: datetime
    ) -> RunResult:
        """Post a run request with the time left as timeout, and classify the answer."""
        timeout = max((deadline_at - datetime.now(UTC)).total_seconds(), 0.001)
        try:
            response = await self._client.post(
                f"{self._origin}{path}", json=payload, timeout=timeout
            )
        except httpx.TransportError as error:
            raise UnknownOutcomeError("analysis dispatch transport failed") from error
        if response.status_code == 429:
            raise ConfirmedRejectionError(
                _retry_after(response.headers.get("retry-after"))
            )
        try:
            body = response.json()
        except ValueError as error:
            # Ray Serve's own 503 (replica queue full, none running): the runner
            # answers JSON, so the request never reached it.
            if response.status_code == 503:
                raise ConfirmedRejectionError(
                    _retry_after(response.headers.get("retry-after"))
                ) from error
            raise UnknownOutcomeError("runner returned an invalid response") from error
        if response.status_code == 200:
            try:
                return RESULT.validate_python(body)
            except ValidationError as error:
                raise UnknownOutcomeError(
                    "runner returned an invalid result"
                ) from error
        try:
            detail = _ERROR_RESPONSE.validate_python(body).detail
        except ValidationError as error:
            raise UnknownOutcomeError(
                f"runner returned HTTP {response.status_code}"
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
