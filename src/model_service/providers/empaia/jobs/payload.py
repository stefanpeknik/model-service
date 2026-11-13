"""The payload JES posts for a job: parsing it, the allowed App Services, credential expiry."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

import jwt
from pydantic import Field, ValidationError

from model_service.providers.empaia.app_service.gateway import EmpaiaJobContext
from model_service.scheduler.public import InvalidScheduleRequestError


if TYPE_CHECKING:
    from model_service.contracts.json_values import JsonValue


class EmpaiaSchedulePayload(EmpaiaJobContext):
    """Require an installation ID alongside the job origin, ID, and private credential."""

    installation_id: Annotated[str, Field(min_length=1)]


def allowed(api_url: str, allowed: tuple[str, ...]) -> bool:
    """Check whether an API URL equals or lies below an allowed prefix.

    Trailing slashes are ignored, and a prefix matches whole path segments only.

    Args:
        api_url: The job's App Service URL.
        allowed: The allowed prefixes, without trailing slashes.

    Returns:
        True if the URL is allowed.
    """
    url = api_url.rstrip("/")
    return any(url == prefix or url.startswith(prefix + "/") for prefix in allowed)


def schedule_payload(values: dict[str, JsonValue]) -> EmpaiaSchedulePayload:
    """Validate provider admission fields, hiding credential-bearing validation input from public errors.

    Args:
        values: The schedule request body.

    Returns:
        The schedule payload.

    Raises:
        InvalidScheduleRequestError: If the body is not a valid payload.
    """
    try:
        return EmpaiaSchedulePayload.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA schedule payload is invalid"
        ) from error


def job_context(values: dict[str, JsonValue]) -> EmpaiaJobContext:
    """Validate a stored protected job context.

    Args:
        values: The stored protected context.

    Returns:
        The job context.

    Raises:
        InvalidScheduleRequestError: If the context is malformed; the message
            holds no context values.
    """
    try:
        return EmpaiaJobContext.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA protected job context is invalid"
        ) from error


def credential_expiry(credential: str) -> datetime:
    """Read the numeric `exp` claim of the credential to cap deadlines.

    This parses the JWT payload without verifying its signature.

    Args:
        credential: The job's token.

    Returns:
        The expiry, in UTC.

    Raises:
        InvalidScheduleRequestError: If the credential is not a JWT or its `exp`
            is missing, not a finite number, or out of range.
    """
    try:
        claims = jwt.decode(
            credential,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "require": ["exp"],
            },
        )
        expires_at = claims["exp"]
        if (
            not isinstance(expires_at, int | float)
            or isinstance(expires_at, bool)
            or not math.isfinite(expires_at)
        ):
            raise TypeError
        return datetime.fromtimestamp(expires_at, tz=UTC)
    except (
        jwt.InvalidTokenError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        OSError,
    ) as error:
        raise InvalidScheduleRequestError(
            "EMPAIA credential does not contain a valid expiry"
        ) from error
