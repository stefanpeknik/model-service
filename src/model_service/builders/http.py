"""Internal HTTP contract between callers and Ray builder replicas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from fastapi import HTTPException, status
from pydantic import Field, StrictBool, TypeAdapter

from model_service.builders.contracts import (
    BuilderFailure,
    BuilderResult,
    ContractViolation,
    InvalidInvocation,
    Invocation,
    ModelBinding,
)


if TYPE_CHECKING:
    from model_service.builders.application import BuilderApplication


@dataclass(frozen=True, slots=True)
class RunRequest:
    """The invocation plus the model resolved for it."""

    invocation: Invocation
    model: ModelBinding


RUN_REQUEST: TypeAdapter[RunRequest] = TypeAdapter(RunRequest)


class ErrorCode(StrEnum):
    INVALID_INVOCATION = "invalid_builder_invocation"
    UNAVAILABLE = "builder_unavailable"
    CONTRACT_VIOLATION = "builder_contract_violation"


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: Annotated[str, Field(strict=True, min_length=1)]
    message: Annotated[str, Field(strict=True, min_length=1)]
    retryable: StrictBool


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    detail: ErrorDetail


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
        request: Invocation and model of the run.

    Returns:
        The builder's result.

    Raises:
        HTTPException: 422 for an invalid invocation or a non-retryable failure,
            500 for a contract violation, 503 for a retryable failure.
    """
    try:
        return await application.run(request.invocation, request.model)
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
