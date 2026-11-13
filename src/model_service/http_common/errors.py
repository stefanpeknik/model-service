"""The public API's error envelope: `{code, message, retryable}`, and how errors map to it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fastapi import HTTPException

from model_service.contracts.failures import UnknownBuilderError, UnknownModelError


class ApiErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_BUILDER = "unknown_builder"
    UNKNOWN_MODEL = "unknown_model"
    INCOMPATIBLE_MODEL = "incompatible_model"
    BUILDER_BACKPRESSURE = "builder_backpressure"
    BUILDER_UNAVAILABLE = "builder_unavailable"
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    detail: ErrorDetail


ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (404, 422, 429, 503)
}


def lookup_error(error: Exception) -> HTTPException:
    """Map a builder or model lookup failure to an HTTP error.

    Args:
        error: An `UnknownBuilderError`, `UnknownModelError` or
            `IncompatibleModelError`.

    Returns:
        404 for an unknown builder or model; otherwise 422 for an incompatible model.
    """
    if isinstance(error, UnknownBuilderError):
        return http_error(404, ApiErrorCode.UNKNOWN_BUILDER, "builder does not exist")
    if isinstance(error, UnknownModelError):
        return http_error(404, ApiErrorCode.UNKNOWN_MODEL, "model is not deployed")
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
