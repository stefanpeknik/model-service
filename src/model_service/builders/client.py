"""Call Ray builder replicas over HTTP and classify the outcome.

A 429 is a confirmed rejection before work started. Transport failures and
malformed responses leave the outcome unknown, because the builder may have run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol
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


class BuilderDispatch(Protocol):
    async def run(
        self,
        builder_id: str,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext,
    ) -> BuilderResult: ...


class ConfirmedRejectionError(RuntimeError):
    def __init__(self, retry_after: timedelta | None = None) -> None:
        super().__init__("builder rejected the request before starting it")
        self.retry_after = retry_after


class UnknownOutcomeError(RuntimeError):
    pass


class RemoteFailureError(RuntimeError):
    def __init__(self, *, code: str, safe_message: str, retryable: bool) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable


class HTTPBuilderDispatch:
    def __init__(self, client: httpx.AsyncClient, *, origin: str) -> None:
        self._client = client
        self._origin = origin.rstrip("/")

    async def run(
        self,
        builder_id: str,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext,
    ) -> BuilderResult:
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
    try:
        seconds = int(value) if value is not None else -1
    except ValueError:
        return None
    return timedelta(seconds=seconds) if seconds >= 0 else None
