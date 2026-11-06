"""Shared error shapes and admission for the public routes.

Routes translate application errors into safe `{code, message, retryable}`
responses.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from fastapi import Header, HTTPException

from model_service.builders.contracts import (
    IncompatibleModelError,
    UnknownBuilderError,
    UnknownModelError,
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

    from model_service.scheduler.contracts import (
        ExecutionView,
        InferenceResult,
        ScheduleRequest,
        ScheduleResult,
    )


logger = logging.getLogger(__name__)


class ApiErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_BUILDER = "unknown_builder"
    UNKNOWN_MODEL = "unknown_model"
    INCOMPATIBLE_MODEL = "incompatible_model"
    SCHEDULE_CONFLICT = "schedule_conflict"
    SCHEDULING_UNAVAILABLE = "scheduling_unavailable"
    UNKNOWN_EXECUTION = "unknown_execution"
    BUILDER_BACKPRESSURE = "builder_backpressure"
    BUILDER_UNAVAILABLE = "builder_unavailable"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    RESULT_UNAVAILABLE = "result_unavailable"
    MODEL_UNAVAILABLE = "model_inference_unavailable"
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
    code: {"model": ErrorResponse} for code in (404, 409, 413, 422, 429, 503)
}


class Scheduler(Protocol):
    @property
    def alive(self) -> bool:
        """Whether the scheduler's worker is still running."""
        ...

    async def check_ready(self) -> None:
        """Check that durable storage is reachable.

        Raises:
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Admit a durable builder execution, or return the one its idempotency key names.

        Args:
            request: The builder, model, provider, body and idempotency key.

        Returns:
            The execution and whether it was created.

        Raises:
            UnknownBuilderError: If the builder is not registered.
            UnknownModelError: If the model is not deployed.
            IncompatibleModelError: If the builder does not accept the model.
            ScheduleConflictError: If the idempotency key names a different request.
            InvalidScheduleRequestError: If the request is invalid.
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def schedule_inference(
        self, request: InferenceScheduleRequest
    ) -> ScheduleResult:
        """Admit a durable raw inference execution, or return the one its key names.

        Args:
            request: The model, payload, headers and idempotency key.

        Returns:
            The execution and whether it was created.

        Raises:
            UnknownModelError: If the model is not deployed.
            IncompatibleInputError: If the payload is not a valid tile for the model.
            ScheduleConflictError: If the idempotency key names a different request.
            InvalidScheduleRequestError: If the request is invalid.
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def get(self, execution_id: str) -> ExecutionView:
        """Read one execution.

        Args:
            execution_id: The execution ID.

        Returns:
            The execution's public view.

        Raises:
            UnknownExecutionError: If no execution has this ID.
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def get_many(self, execution_ids: list[str]) -> list[ExecutionView]:
        """Read the listed executions that exist.

        Args:
            execution_ids: The execution IDs.

        Returns:
            Views of the existing executions; unknown IDs are left out.

        Raises:
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def inference_result(
        self, execution_id: str
    ) -> tuple[InferenceResult, bytes]:
        """Read the stored model response of a succeeded inference execution.

        Args:
            execution_id: The execution ID.

        Returns:
            The response's media type and headers, and its body.

        Raises:
            UnknownExecutionError: If no execution has this ID.
            ResultUnavailableError: If the execution has no stored response.
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def cancel(self, execution_id: str) -> ExecutionView:
        """Cancel an execution.

        Args:
            execution_id: The execution ID.

        Returns:
            The execution's view after the cancellation request.

        Raises:
            UnknownExecutionError: If no execution has this ID.
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...


class Capacity(ABC):
    @abstractmethod
    async def explain(self, execution: ExecutionView) -> ExecutionView:
        """Add a message explaining why a running execution makes no progress.

        Args:
            execution: The execution to explain.

        Returns:
            The execution, with a message if one applies.
        """
        ...


class NoCapacityInfo(Capacity):
    async def explain(self, execution: ExecutionView) -> ExecutionView:
        """Leave the execution unexplained, for deployments without a Ray dashboard.

        Args:
            execution: The execution.

        Returns:
            The execution unchanged.
        """
        return execution


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
