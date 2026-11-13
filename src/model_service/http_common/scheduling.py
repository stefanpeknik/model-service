"""HTTP plumbing for scheduling routes: the Idempotency-Key header, and scheduler refusals as HTTP errors (public and provider routes)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastapi import Header

from model_service.contracts.failures import (
    IncompatibleModelError,
    UnknownBuilderError,
    UnknownModelError,
)
from model_service.http_common.errors import (
    ApiErrorCode,
    http_error,
    lookup_error,
    unavailable,
)
from model_service.scheduler.public import (
    InvalidScheduleRequestError,
    ScheduleConflictError,
    SchedulingUnavailableError,
)


if TYPE_CHECKING:
    from fastapi import Response

    from model_service.scheduler.protocols import Scheduler
    from model_service.scheduler.public import ScheduleRequest, ScheduleResult


logger = logging.getLogger(__name__)


def require_idempotency_key(
    idempotency_key: Annotated[str | None, Header(min_length=1)] = None,
) -> str:
    """Require the `Idempotency-Key` header that names a durable submission.

    Args:
        idempotency_key: The header value, or None if absent.

    Returns:
        The idempotency key.

    Raises:
        HTTPException: 422 if the header is missing.
    """
    if idempotency_key is None:
        raise http_error(
            422, ApiErrorCode.INVALID_REQUEST, "Idempotency-Key header is required"
        )
    return idempotency_key


async def submit(
    scheduler: Scheduler,
    request: ScheduleRequest,
    response: Response,
) -> ScheduleResult:
    """Admit a builder execution and map refusals to API errors.

    Refusals are logged, since callers may not log the response body.

    Args:
        scheduler: Admits the execution.
        request: The builder schedule request.
        response: The outgoing response; its status becomes 202 when the
            execution is created and 200 when an existing one is returned.

    Returns:
        The execution and whether it was created.

    Raises:
        HTTPException: 404 for an unknown builder or model, 409 for an
            idempotency conflict, 422 for an incompatible model or invalid
            request, 503 if scheduling is unavailable.
    """
    try:
        result = await scheduler.schedule(request)
    except (UnknownBuilderError, UnknownModelError, IncompatibleModelError) as error:
        _log_refusal(request, error)
        raise lookup_error(error) from error
    except ScheduleConflictError as error:
        _log_refusal(request, error)
        raise http_error(409, ApiErrorCode.SCHEDULE_CONFLICT, str(error)) from error
    except InvalidScheduleRequestError as error:
        _log_refusal(request, error)
        raise http_error(422, ApiErrorCode.INVALID_REQUEST, str(error)) from error
    except SchedulingUnavailableError as error:
        _log_refusal(request, error)
        raise unavailable(error) from error
    response.status_code = 202 if result.created else 200
    return result


def _log_refusal(request: ScheduleRequest, error: Exception) -> None:
    """Log why a schedule request was refused; error messages are the safe public ones."""
    subject = f"{request.provider} {request.builder_id} on {request.model_id}"
    logger.warning("Refused %s: %s", subject, error)
