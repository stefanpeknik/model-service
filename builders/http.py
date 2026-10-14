"""Internal HTTP contract between the facade/scheduler and Ray builder replicas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from fastapi import HTTPException, status
from pydantic import Field, StrictBool, TypeAdapter

from builders.contracts import (
    BuilderFailure,
    BuilderResult,
    ContractViolation,
    ExecutionContext,
    InvalidInvocation,
    Invocation,
    ModelBinding,
)


if TYPE_CHECKING:
    from builders.application import BuilderApplication


@dataclass(frozen=True, slots=True)
class RunRequest:
    """The stored invocation plus the model resolved for this attempt."""

    invocation: Invocation
    model: ModelBinding
    execution_context: ExecutionContext = field(default_factory=ExecutionContext)


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
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "retryable": retryable},
    )


async def run_http(
    application: BuilderApplication, request: RunRequest
) -> BuilderResult:
    """Translate builder exceptions into the safe internal error envelope."""
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
