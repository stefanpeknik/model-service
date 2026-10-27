"""Shared error shapes and admission for the public routes.

Routes translate application errors into safe `{code, message, retryable}`
responses.
"""

from __future__ import annotations

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
    async def check_ready(self) -> None: ...

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult: ...

    async def schedule_inference(
        self, request: InferenceScheduleRequest
    ) -> ScheduleResult: ...

    async def get(self, execution_id: str) -> ExecutionView: ...

    async def get_many(self, execution_ids: list[str]) -> list[ExecutionView]: ...

    async def inference_result(
        self, execution_id: str
    ) -> tuple[InferenceResult, bytes]: ...

    async def cancel(self, execution_id: str) -> ExecutionView: ...


class Capacity(Protocol):
    async def explain(self, execution: ExecutionView) -> ExecutionView: ...


class NoCapacityInfo:
    async def explain(self, execution: ExecutionView) -> ExecutionView:
        return execution


def require_idempotency_key(
    idempotency_key: Annotated[str | None, Header(min_length=1)] = None,
) -> str:
    """Require the `Idempotency-Key` header that names a durable submission."""
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
        raise lookup_error(error) from error
    except ScheduleConflictError as error:
        raise http_error(409, ApiErrorCode.SCHEDULE_CONFLICT, str(error)) from error
    except (InvalidScheduleRequestError, IncompatibleInputError) as error:
        raise http_error(422, ApiErrorCode.INVALID_REQUEST, str(error)) from error
    except SchedulingUnavailableError as error:
        raise unavailable(error) from error
    response.status_code = 202 if result.created else 200
    return result


def unavailable(error: SchedulingUnavailableError) -> HTTPException:
    return http_error(
        503, ApiErrorCode.SCHEDULING_UNAVAILABLE, str(error), retryable=True
    )


def lookup_error(error: Exception) -> HTTPException:
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
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "retryable": retryable},
        headers=headers,
    )
