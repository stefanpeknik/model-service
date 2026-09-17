from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import httpx

from workflow_execution.contracts import invocation_to_dict, result_from_dict


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from workflow_execution.contracts import WorkflowInvocation, WorkflowResult


class WorkflowDispatch(Protocol):
    async def run(
        self,
        workflow_id: str,
        processing_attempt_id: str,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult: ...


class ConfirmedDispatchRejectionError(RuntimeError):
    def __init__(self, retry_after: timedelta | None = None) -> None:
        super().__init__("workflow dispatch was rejected before acceptance")
        self.retry_after = retry_after


class UnknownDispatchOutcomeError(RuntimeError):
    pass


class RemoteWorkflowFailureError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        safe_message: str,
        retryable: bool,
    ) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable


class HTTPResponse(Protocol):
    status_code: int

    @property
    def headers(self) -> Mapping[str, str]: ...

    def json(self) -> object: ...


class HTTPClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: object,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HTTPResponse: ...


class HTTPWorkflowDispatch:
    def __init__(self, client: HTTPClient, *, origin: str) -> None:
        self._client = client
        self._origin = origin.rstrip("/")

    async def run(
        self,
        workflow_id: str,
        processing_attempt_id: str,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult:
        timeout = max(
            (invocation.deadline_at - datetime.now(UTC)).total_seconds(),
            0.001,
        )
        try:
            response = await self._client.post(
                f"{self._origin}/workflows/{workflow_id}/",
                json=invocation_to_dict(invocation),
                headers={"x-processing-attempt-id": processing_attempt_id},
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
    def __init__(
        self,
        handlers: Mapping[
            str,
            Callable[[WorkflowInvocation], WorkflowResult | Awaitable[WorkflowResult]],
        ],
    ) -> None:
        self._handlers = dict(handlers)

    async def run(
        self,
        workflow_id: str,
        _processing_attempt_id: str,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult:
        value = self._handlers[workflow_id](invocation)
        if inspect.isawaitable(value):
            return await value
        return value
