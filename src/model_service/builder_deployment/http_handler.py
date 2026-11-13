"""Answer a builder run request and map exceptions to the deployment's HTTP errors.

The format is contracts/builder_http.py; the public API's is http_common/errors.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from model_service.contracts.builder_http import (
    ErrorCode,
    RunRequest,
)
from model_service.contracts.failures import (
    BuilderFailure,
    ContractViolation,
    InvalidInvocation,
)
from model_service.contracts.result import BuilderResult


if TYPE_CHECKING:
    from model_service.builder_deployment.runner import BuilderApplication


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


async def run_http(
    application: BuilderApplication, request: RunRequest
) -> BuilderResult:
    """Run an invocation and translate builder exceptions into the safe error envelope.

    Args:
        application: Builder application to run.
        request: Invocation, model and execution context of the attempt.

    Returns:
        The builder's result.

    Raises:
        HTTPException: 422 for an invalid invocation or a non-retryable failure,
            500 for a contract violation, 503 for a retryable failure.
    """
    try:
        return await application.run(
            request.invocation, request.model, request.execution_context
        )
    except InvalidInvocation as exc:
        raise error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_INVOCATION,
            str(exc),
            retryable=False,
        ) from exc
    except ContractViolation as exc:
        raise error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.CONTRACT_VIOLATION,
            "deployed builder violated its contract",
            retryable=False,
        ) from exc
    except BuilderFailure as exc:
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
                "message": "builder invocation is invalid",
                "retryable": False,
            }
        },
    )
