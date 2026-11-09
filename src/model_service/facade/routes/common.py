"""Admission shared by the public routes: the idempotency key and scheduling refusals."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastapi import Header

from model_service.builders.contracts import (
    IncompatibleModelError,
    UnknownBuilderError,
    UnknownModelError,
)
from model_service.facade.errors import (
    ApiErrorCode,
    http_error,
    lookup_error,
    unavailable,
)
from model_service.scheduler.contracts import (
    InferenceScheduleRequest,
    InvalidScheduleRequestError,
    ScheduleConflictError,
    SchedulingUnavailableError,
)
from model_service.scheduler.inference import IncompatibleInputError


if TYPE_CHECKING:
    from fastapi import Response

    from model_service.facade.ports import Scheduler
    from model_service.scheduler.contracts import ScheduleRequest, ScheduleResult


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
    request: ScheduleRequest | InferenceScheduleRequest,
    response: Response,
) -> ScheduleResult:
    """Admit a builder or inference execution and map refusals to API errors.

    Refusals are logged, since callers may not log the response body.

    Args:
        scheduler: Admits the execution.
        request: A builder or raw inference schedule request.
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
        if isinstance(request, InferenceScheduleRequest):
            result = await scheduler.schedule_inference(request)
        else:
            result = await scheduler.schedule(request)
    except (
        UnknownBuilderError,
        UnknownModelError,
        IncompatibleModelError,
    ) as error:
        _log_refusal(request, error)
        raise lookup_error(error) from error
    except ScheduleConflictError as error:
        _log_refusal(request, error)
        raise http_error(409, ApiErrorCode.SCHEDULE_CONFLICT, str(error)) from error
    except (InvalidScheduleRequestError, IncompatibleInputError) as error:
        _log_refusal(request, error)
        raise http_error(422, ApiErrorCode.INVALID_REQUEST, str(error)) from error
    except SchedulingUnavailableError as error:
        _log_refusal(request, error)
        raise unavailable(error) from error
    response.status_code = 202 if result.created else 200
    return result


def _log_refusal(
    request: ScheduleRequest | InferenceScheduleRequest, error: Exception
) -> None:
    """Log why a schedule request was refused.

    Args:
        request: The refused request.
        error: The refusal; its message is safe to log.
    """
    # The callers (JES among them) may not log the response body, so the
    # reason is kept here; error messages are the safe public ones.
    if isinstance(request, InferenceScheduleRequest):
        subject = f"inference on {request.model_id}"
    else:
        subject = f"{request.provider} {request.builder_id} on {request.model_id}"
    logger.warning("Refused %s: %s", subject, error)
