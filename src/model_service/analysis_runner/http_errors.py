"""Answer a run request and map exceptions to the runner's HTTP error envelope.

The format is in contracts/analysis_http.py; the public API's is api/http/errors.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from model_service.contracts.analysis_http import ErrorCode
from model_service.contracts.failures import InvalidInvocation, RunFailure


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from model_service.contracts.result import RunResult


def error(
    status_code: int, code: str, message: str, *, retryable: bool
) -> HTTPException:
    """Build the internal error envelope as an HTTP exception.

    Args:
        status_code: HTTP status to answer with.
        code: Stable error code.
        message: Message safe to show callers.
        retryable: Whether a retry may succeed.

    Returns:
        The exception for FastAPI to answer with.
    """
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "retryable": retryable},
    )


async def answer_http(run: Awaitable[RunResult]) -> RunResult:
    """Await a run and translate its exceptions into the safe error envelope.

    Args:
        run: The run of one request.

    Returns:
        The run's result.

    Raises:
        HTTPException: 422 for an invalid invocation or a non-retryable failure,
            503 for a retryable failure.
    """
    try:
        return await run
    except InvalidInvocation as exc:
        raise error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_INVOCATION,
            str(exc),
            retryable=False,
        ) from exc
    except RunFailure as exc:
        raise error(
            status.HTTP_503_SERVICE_UNAVAILABLE
            if exc.retryable
            else status.HTTP_422_UNPROCESSABLE_CONTENT,
            exc.code,
            exc.safe_message,
            retryable=exc.retryable,
        ) from exc


async def invalid_request(
    _request: Request, _error: RequestValidationError
) -> JSONResponse:
    """Answer a malformed request without echoing it, since it may hold credentials.

    Args:
        _request: The rejected request.
        _error: The validation error.

    Returns:
        A 422 response with the invalid-invocation envelope.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": {
                "code": ErrorCode.INVALID_INVOCATION,
                "message": "analysis run request is invalid",
                "retryable": False,
            }
        },
    )
