"""The public API's error envelope: `{code, message, retryable}`, and how errors map to it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fastapi import HTTPException


class ApiErrorCode(StrEnum):
    UNKNOWN_MODEL = "unknown_model"


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    detail: ErrorDetail


ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (404,)
}


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
