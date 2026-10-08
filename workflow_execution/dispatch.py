"""Dispatch typed workflows over HTTP and classify rejection, failure, and uncertain outcomes.

A confirmed 429 was rejected before acceptance. Transport failures or malformed
responses can follow accepted remote work, so their outcomes remain unknown.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import httpx

from workflow_execution.http import WorkflowRunRequest, run_request_to_dict
from workflow_execution.serialization import result_from_dict


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from workflow_execution.contracts import (
        WorkflowExecutionContext,
        WorkflowInvocation,
        WorkflowResult,
    )


class WorkflowDispatch(Protocol):
    """Define transport-independent invocation of a workflow with private execution context."""

    async def run(
        self,
        workflow_id: str,
        execution_context: WorkflowExecutionContext,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult:
        """Invoke the selected workflow and return outputs or a classified dispatch failure.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            execution_context: Private attempt identity and provider context kept
                separate from the invocation.
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.

        Raises:
            ConfirmedDispatchRejectionError: The remote deployment confirms rejection
                before accepting work.
            RemoteWorkflowFailureError: The remote workflow returns a safe structured
                failure.
            UnknownDispatchOutcomeError: Transport or response decoding cannot establish
                the remote outcome.
        """
        ...


class ConfirmedDispatchRejectionError(RuntimeError):
    """Signal definite rejection before work was accepted, optionally with a retry delay."""

    def __init__(self, retry_after: timedelta | None = None) -> None:
        """Capture the minimum retry delay supplied by the rejecting deployment.

        Args:
            retry_after: Minimum delay before retrying the rejected request.
        """
        super().__init__("workflow dispatch was rejected before acceptance")
        self.retry_after = retry_after


class UnknownDispatchOutcomeError(RuntimeError):
    """Signal that accepted remote work cannot be ruled out or its result cannot be decoded."""


class RemoteWorkflowFailureError(RuntimeError):
    """Carry a structured safe failure reported by the remote workflow application."""

    def __init__(
        self,
        *,
        code: str,
        safe_message: str,
        retryable: bool,
    ) -> None:
        """Store the remote safe code, message, and retry permission.

        Args:
            code: Stable failure identifier, or HTTP status when classifying an upstream
                response.
            safe_message: Message permitted in public errors without internal details or
                credentials.
            retryable: Whether another attempt is permitted for this classified failure.
        """
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable


class HTTPResponse(Protocol):
    """Specify the response fields needed to classify workflow dispatch outcomes.

    Attributes:
        status_code: HTTP status of the upstream response.
    """

    status_code: int

    @property
    def headers(self) -> Mapping[str, str]:
        """Expose response headers, including any Retry-After value.

        Returns:
            Mapping[str, str]: HTTP response headers, including Retry-After when
                supplied.
        """
        ...

    def json(self) -> object:
        """Decode the response body for typed result or structured failure parsing.

        Returns:
            object: Decoded response JSON for workflow output or failure parsing.

        Raises:
            ValueError: The HTTP response body cannot be decoded as JSON.
        """
        ...


class HTTPClient(Protocol):
    """Specify the async POST boundary used by workflow dispatch and test clients."""

    async def post(
        self,
        url: str,
        *,
        json: object,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HTTPResponse:
        """Send a JSON request with headers and a bounded timeout, returning a response.

        Args:
            url: Complete upstream HTTP request URL.
            json: JSON-compatible workflow request passed to the HTTP client.
            headers: HTTP metadata to filter, send, or include in an error response.
            timeout: Request or scenario time budget in seconds.

        Returns:
            HTTPResponse: HTTP response used to classify workflow dispatch acceptance
                and outcome.
        """
        ...


class HTTPWorkflowDispatch:
    """Send workflow invocations to Ray Serve's internal workflow HTTP routes."""

    def __init__(self, client: HTTPClient, *, origin: str) -> None:
        """Retain the caller-owned HTTP client and normalize the dispatch origin.

        Args:
            client: Caller-owned HTTP client; this adapter does not close it.
            origin: Upstream service HTTP origin, normalized without a trailing slash.
        """
        self._client = client
        self._origin = origin.rstrip("/")

    async def run(
        self,
        workflow_id: str,
        execution_context: WorkflowExecutionContext,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult:
        """POST the internal request using the remaining deadline and classify its result.

        A 429 becomes confirmed rejection. Transport errors, invalid results, or
        unrecognized responses become unknown outcomes; structured remote failures
        retain their retry decision.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            execution_context: Private attempt identity and provider context kept
                separate from the invocation.
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.

        Raises:
            ConfirmedDispatchRejectionError: The remote deployment confirms rejection
                before accepting work.
            RemoteWorkflowFailureError: The remote workflow returns a safe structured
                failure.
            UnknownDispatchOutcomeError: Transport or response decoding cannot establish
                the remote outcome.
            asyncio.CancelledError: The owning task, caller, deadline, or lease monitor
                cancels the operation.
        """
        timeout = max(
            (invocation.deadline_at - datetime.now(UTC)).total_seconds(),
            0.001,
        )
        try:
            response = await self._client.post(
                f"{self._origin}/workflows/{workflow_id}/",
                json=run_request_to_dict(
                    WorkflowRunRequest.from_invocation(invocation, execution_context)
                ),
                headers=(
                    {"x-processing-attempt-id": execution_context.processing_attempt_id}
                    if execution_context.processing_attempt_id is not None
                    else {}
                ),
                timeout=timeout,
            )
        except httpx.TransportError as error:
            raise UnknownDispatchOutcomeError(
                "workflow dispatch transport failed"
            ) from error
        if response.status_code == 429:
            raise ConfirmedDispatchRejectionError(
                _retry_after(response.headers.get("retry-after"))
            )
        try:
            body = response.json()
        except (TypeError, ValueError) as error:
            raise UnknownDispatchOutcomeError(
                "workflow dispatch returned an invalid response"
            ) from error
        if response.status_code == 200:
            try:
                return result_from_dict(body)
            except (TypeError, ValueError) as error:
                raise UnknownDispatchOutcomeError(
                    "workflow dispatch returned an invalid result"
                ) from error
        failure = _remote_failure(body)
        if failure is not None:
            raise failure
        raise UnknownDispatchOutcomeError(
            f"workflow dispatch returned HTTP {response.status_code}"
        )


def _retry_after(value: str | None) -> timedelta | None:
    """Parse a nonnegative integer Retry-After value as a duration, ignoring invalid values.

    Args:
        value: Retry-After header as integer seconds or an HTTP-date, or None when
            absent.

    Returns:
        timedelta | None: Nonnegative delay parsed from the header, or None for an
            absent or invalid header.
    """
    if value is None:
        return None
    try:
        seconds = int(value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return timedelta(seconds=seconds)


def _remote_failure(body: object) -> RemoteWorkflowFailureError | None:
    """Decode a strictly typed structured failure or return None if the response lacks the expected fields.

    Args:
        body: Typed or JSON request payload accepted by this boundary.

    Returns:
        RemoteWorkflowFailureError | None: Safe structured remote failure, or None if
            response fields do not match the failure contract.
    """
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    if not isinstance(detail, dict):
        return None
    code = detail.get("code")
    message = detail.get("message")
    retryable = detail.get("retryable")
    if (
        not isinstance(code, str)
        or not code
        or not isinstance(message, str)
        or not message
        or not isinstance(retryable, bool)
    ):
        return None
    return RemoteWorkflowFailureError(
        code=code,
        safe_message=message,
        retryable=retryable,
    )


class InMemoryWorkflowDispatch:
    """Invoke registered local workflow handlers for tests without HTTP or Ray."""

    def __init__(
        self,
        handlers: Mapping[
            str,
            Callable[
                [WorkflowInvocation, WorkflowExecutionContext],
                WorkflowResult | Awaitable[WorkflowResult],
            ],
        ],
    ) -> None:
        """Copy the workflow ID to handler mapping for local dispatch.

        Args:
            handlers: Workflow ID to local computation handler mapping.
        """
        self._handlers = dict(handlers)

    async def run(
        self,
        workflow_id: str,
        execution_context: WorkflowExecutionContext,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult:
        """Call a local handler and await its result if needed; unknown IDs raise a mapping lookup error.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            execution_context: Private attempt identity and provider context kept
                separate from the invocation.
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.
        """
        value = self._handlers[workflow_id](invocation, execution_context)
        if inspect.isawaitable(value):
            return await value
        return value
