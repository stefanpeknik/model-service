"""Synchronous runs over HTTP: wait while the caller stays connected, answer safe errors.

Builder runs and analysis runs share this: lookups and admission errors are
answered before dispatch, and what the replica answered is mapped to the
API's envelope.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from model_service.api.http.errors import ApiErrorCode, http_error, lookup_error
from model_service.api.scheduler.builder_dispatch import (
    ConfirmedRejectionError,
    RemoteFailureError,
    UnknownOutcomeError,
)
from model_service.contracts.failures import (
    IncompatibleModelError,
    InvalidInvocation,
    UnknownAnalysisError,
    UnknownBuilderError,
    UnknownModelError,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from fastapi import HTTPException, Request

    from model_service.contracts.result import BuilderResult


def runs_unavailable(what: str) -> HTTPException:
    """The 503 answered when the API has no runner for synchronous runs.

    Args:
        what: What isn't available, e.g. `synchronous builder runs`.

    Returns:
        The HTTP error to raise.
    """
    return http_error(
        503, ApiErrorCode.BUILDER_UNAVAILABLE, f"{what} are unavailable", retryable=True
    )


async def run_synchronously(
    request: Request, operation: Awaitable[BuilderResult]
) -> BuilderResult:
    """Await a run while the caller stays connected, mapping its failures.

    Args:
        request: The HTTP request, watched for disconnects.
        operation: The run: admission, then dispatch.

    Returns:
        The run's result.

    Raises:
        HTTPException: 404 for an unknown builder, model or analysis; 422 for
            an incompatible model, an invalid request or a non-retryable
            failure; 429 when the replica has no capacity; 503 for a
            retryable failure or an unknown outcome.
        asyncio.CancelledError: If the client disconnects first.
    """
    try:
        return await _run_while_connected(request, operation)
    except (
        UnknownBuilderError,
        UnknownModelError,
        UnknownAnalysisError,
        IncompatibleModelError,
    ) as error:
        raise lookup_error(error) from error
    except InvalidInvocation as error:
        raise http_error(422, ApiErrorCode.INVALID_REQUEST, str(error)) from error
    except ConfirmedRejectionError as error:
        headers = None
        if error.retry_after is not None:
            headers = {"Retry-After": str(int(error.retry_after.total_seconds()))}
        raise http_error(
            429,
            ApiErrorCode.BUILDER_BACKPRESSURE,
            "builder capacity is temporarily unavailable",
            retryable=True,
            headers=headers,
        ) from error
    except RemoteFailureError as error:
        raise http_error(
            503 if error.retryable else 422,
            error.code,
            error.safe_message,
            retryable=error.retryable,
        ) from error
    except UnknownOutcomeError as error:
        raise http_error(
            503,
            ApiErrorCode.BUILDER_UNAVAILABLE,
            "builder outcome is unknown",
            retryable=True,
        ) from error


async def _run_while_connected[T](request: Request, operation: Awaitable[T]) -> T:
    """Await the operation, cancelling it if the client disconnects first.

    Args:
        request: The request whose connection is watched.
        operation: The work to run.

    Returns:
        The operation's result.

    Raises:
        asyncio.CancelledError: If the client disconnects before the operation ends.
    """
    operation_task = asyncio.ensure_future(operation)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _pending = await asyncio.wait(
            {operation_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if operation_task in done:
            return await operation_task
        raise asyncio.CancelledError
    finally:
        for task in (operation_task, disconnect_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(operation_task, disconnect_task, return_exceptions=True)


async def _wait_for_disconnect(request: Request) -> None:
    """Return once the client disconnects.

    Args:
        request: The request whose ASGI messages are read.
    """
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return
