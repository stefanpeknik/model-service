"""The public API's error envelope: `{code, message, retryable}`, and how errors map to it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from model_service.contracts.failures import (
    UnknownAnalysisError,
    UnknownBuilderError,
    UnknownModelError,
)


if TYPE_CHECKING:
    from model_service.api.scheduler.public import SchedulingUnavailableError


class ApiErrorCode(StrEnum):
    """Error codes the API answers with."""

    INVALID_REQUEST = "invalid_request"
    UNKNOWN_BUILDER = "unknown_builder"
    UNKNOWN_MODEL = "unknown_model"
    UNKNOWN_ANALYSIS = "unknown_analysis"
    INCOMPATIBLE_MODEL = "incompatible_model"
    SCHEDULE_CONFLICT = "schedule_conflict"
    SCHEDULING_UNAVAILABLE = "scheduling_unavailable"
    UNKNOWN_EXECUTION = "unknown_execution"
    BUILDER_BACKPRESSURE = "builder_backpressure"
    BUILDER_UNAVAILABLE = "builder_unavailable"
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    """An API error.

    Attributes:
        code: Machine-readable error code.
        message: Message safe to show callers.
        retryable: Whether the same request may succeed later.
    """

    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    """The API's error body.

    Attributes:
        detail: The error.
    """

    detail: ErrorDetail


ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (404, 409, 422, 429, 503)
}


def unavailable(error: SchedulingUnavailableError) -> HTTPException:
    """Map a storage outage to a retryable 503.

    Args:
        error: The scheduling outage.

    Returns:
        The HTTP error to raise.
    """
    return http_error(
        503, ApiErrorCode.SCHEDULING_UNAVAILABLE, str(error), retryable=True
    )


def lookup_error(error: Exception) -> HTTPException:
    """Map a builder, model or analysis lookup failure to an HTTP error.

    Args:
        error: An `UnknownBuilderError`, `UnknownModelError`,
            `UnknownAnalysisError` or `IncompatibleModelError`.

    Returns:
        404 for an unknown builder, model or analysis; otherwise 422 for an
        incompatible model.
    """
    if isinstance(error, UnknownBuilderError):
        return http_error(404, ApiErrorCode.UNKNOWN_BUILDER, "builder does not exist")
    if isinstance(error, UnknownModelError):
        return http_error(404, ApiErrorCode.UNKNOWN_MODEL, "model is not deployed")
    if isinstance(error, UnknownAnalysisError):
        return http_error(
            404, ApiErrorCode.UNKNOWN_ANALYSIS, "analysis is not deployed"
        )
    return http_error(422, ApiErrorCode.INCOMPATIBLE_MODEL, str(error))


def http_error(
    status_code: int,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    """Build an HTTP error with the API's `{code, message, retryable}` detail.

    Args:
        status_code: The HTTP status.
        code: The API error code.
        message: A message safe to show the caller.
        retryable: Whether the caller may retry the same request.
        headers: Extra response headers.

    Returns:
        The exception to raise.
    """
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "retryable": retryable},
        headers=headers,
    )
