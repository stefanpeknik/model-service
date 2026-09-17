from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import Lifespan

from execution_scheduler.contracts import (
    ExecutionView,
    InvalidScheduleRequestError,
    ScheduleConflictError,
    ScheduleRequest,
    ScheduleResult,
    SchedulingUnavailableError,
    UnknownExecutionError,
    UnknownWorkflowError,
)
from workflow_execution.contracts import JsonValue  # noqa: TC001 - FastAPI needs it.


class ApiErrorCode(StrEnum):
    INVALID_SCHEDULE_REQUEST = "invalid_schedule_request"
    UNKNOWN_WORKFLOW = "unknown_workflow"
    SCHEDULE_CONFLICT = "schedule_conflict"
    SCHEDULING_UNAVAILABLE = "scheduling_unavailable"
    UNKNOWN_EXECUTION = "unknown_execution"
    WORKFLOW_APPLICATION_UNAVAILABLE = "workflow_application_unavailable"


@dataclass(frozen=True, slots=True)
class ScheduleBody:
    provider: str
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: ApiErrorCode
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    detail: ErrorDetail


class ExecutionSchedulerInterface(Protocol):
    async def schedule(self, request: ScheduleRequest) -> ScheduleResult: ...

    async def get(self, execution_id: str) -> ExecutionView: ...

    async def cancel(self, execution_id: str) -> ExecutionView: ...


class DeploymentHealth(Protocol):
    async def missing_workflows(self) -> tuple[str, ...]: ...


def create_execution_app(
    scheduler: ExecutionSchedulerInterface,
    *,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealth | None = None,
) -> FastAPI:
    app = FastAPI(title="model-service execution scheduler", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": {
                    "code": ApiErrorCode.INVALID_SCHEDULE_REQUEST,
                    "message": "request body is invalid",
                    "retryable": False,
                }
            },
        )

    @app.post(
        "/v1/workflows/{workflow_id}/schedule",
        response_model=ScheduleResult,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            200: {"model": ScheduleResult},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def schedule(
        workflow_id: str,
        body: ScheduleBody,
        response: Response,
    ) -> ScheduleResult:
        try:
            result = await scheduler.schedule(
                ScheduleRequest(
                    workflow_id=workflow_id,
                    provider=body.provider,
                    body=body.body,
                    requested_timeout_seconds=body.requested_timeout_seconds,
                    idempotency_key=body.idempotency_key,
                )
            )
        except UnknownWorkflowError as error:
            raise _http_error(
                404,
                ApiErrorCode.UNKNOWN_WORKFLOW,
                "workflow is not declared",
            ) from error
        except ScheduleConflictError as error:
            raise _http_error(
                409,
                ApiErrorCode.SCHEDULE_CONFLICT,
                str(error),
            ) from error
        except InvalidScheduleRequestError as error:
            raise _http_error(
                422,
                ApiErrorCode.INVALID_SCHEDULE_REQUEST,
                str(error),
            ) from error
        except SchedulingUnavailableError as error:
            raise _http_error(
                503,
                ApiErrorCode.SCHEDULING_UNAVAILABLE,
                str(error),
                retryable=True,
            ) from error
        response.status_code = 202 if result.created else 200
        return result

    @app.get(
        "/v1/executions/{execution_id}",
        response_model=ExecutionView,
        responses={404: {"model": ErrorResponse}},
    )
    async def get_execution(execution_id: str) -> ExecutionView:
        try:
            return await scheduler.get(execution_id)
        except UnknownExecutionError as error:
            raise _http_error(
                404,
                ApiErrorCode.UNKNOWN_EXECUTION,
                "execution does not exist",
            ) from error

    @app.post(
        "/v1/executions/{execution_id}/cancel",
        response_model=ExecutionView,
        responses={404: {"model": ErrorResponse}},
    )
    async def cancel_execution(execution_id: str) -> ExecutionView:
        try:
            return await scheduler.cancel(execution_id)
        except UnknownExecutionError as error:
            raise _http_error(
                404,
                ApiErrorCode.UNKNOWN_EXECUTION,
                "execution does not exist",
            ) from error

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readiness() -> dict[str, object]:
        if deployment_health is None:
            return {"status": "ok"}
        missing = await deployment_health.missing_workflows()
        if missing:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": ApiErrorCode.WORKFLOW_APPLICATION_UNAVAILABLE,
                    "message": "declared workflow applications are not deployed",
                    "retryable": True,
                    "workflows": list(missing),
                },
            )
        return {"status": "ok"}

    return app


def _http_error(
    status_code: int,
    code: ApiErrorCode,
    message: str,
    *,
    retryable: bool = False,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "retryable": retryable},
    )
