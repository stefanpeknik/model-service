from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

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
from execution_scheduler.dispatch import (
    ConfirmedDispatchRejectionError,
    RemoteWorkflowFailureError,
    UnknownDispatchOutcomeError,
)
from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    JsonValue,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.http import WorkflowErrorResponse


if TYPE_CHECKING:
    from collections.abc import Awaitable


class ApiErrorCode(StrEnum):
    INVALID_SCHEDULE_REQUEST = "invalid_schedule_request"
    INVALID_WORKFLOW_INVOCATION = "invalid_workflow_invocation"
    UNKNOWN_WORKFLOW = "unknown_workflow"
    SCHEDULE_CONFLICT = "schedule_conflict"
    SCHEDULING_UNAVAILABLE = "scheduling_unavailable"
    UNKNOWN_EXECUTION = "unknown_execution"
    WORKFLOW_BACKPRESSURE = "workflow_backpressure"
    WORKFLOW_DISPATCH_UNAVAILABLE = "workflow_dispatch_unavailable"
    WORKFLOW_APPLICATION_UNAVAILABLE = "workflow_application_unavailable"


@dataclass(frozen=True, slots=True)
class ScheduleBody:
    provider: str
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SchedulerErrorDetail:
    code: ApiErrorCode
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class SchedulerErrorResponse:
    detail: SchedulerErrorDetail


class ExecutionSchedulerInterface(Protocol):
    async def schedule(self, request: ScheduleRequest) -> ScheduleResult: ...

    async def get(self, execution_id: str) -> ExecutionView: ...

    async def cancel(self, execution_id: str) -> ExecutionView: ...


class SynchronousWorkflowInterface(Protocol):
    async def run(
        self,
        workflow_id: str,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult: ...


class DeploymentHealth(Protocol):
    async def missing_workflows(self) -> tuple[str, ...]: ...


def create_execution_app(
    scheduler: ExecutionSchedulerInterface,
    *,
    workflow_runner: SynchronousWorkflowInterface | None = None,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealth | None = None,
) -> FastAPI:
    app = FastAPI(title="model-service execution scheduler", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        is_workflow_run = request.url.path.endswith("/run")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": {
                    "code": (
                        ApiErrorCode.INVALID_WORKFLOW_INVOCATION
                        if is_workflow_run
                        else ApiErrorCode.INVALID_SCHEDULE_REQUEST
                    ),
                    "message": (
                        "workflow invocation is invalid"
                        if is_workflow_run
                        else "request body is invalid"
                    ),
                    "retryable": False,
                }
            },
        )

    @app.post(
        "/v1/workflows/{workflow_id}/run",
        response_model=WorkflowResult,
        responses={
            404: {"model": SchedulerErrorResponse},
            422: {"model": WorkflowErrorResponse},
            429: {"model": SchedulerErrorResponse},
            503: {"model": WorkflowErrorResponse},
        },
    )
    async def run_workflow(
        request: Request,
        workflow_id: str,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult:
        if workflow_runner is None:
            raise _http_error(
                503,
                ApiErrorCode.WORKFLOW_DISPATCH_UNAVAILABLE,
                "synchronous workflow execution is unavailable",
                retryable=True,
            )
        try:
            return await _run_while_connected(
                request,
                workflow_runner.run(workflow_id, invocation),
            )
        except UnknownWorkflowError as error:
            raise _http_error(
                404,
                ApiErrorCode.UNKNOWN_WORKFLOW,
                "workflow is not declared",
            ) from error
        except InvalidWorkflowInvocation as error:
            raise _http_error(
                422,
                ApiErrorCode.INVALID_WORKFLOW_INVOCATION,
                str(error),
            ) from error
        except ConfirmedDispatchRejectionError as error:
            headers = None
            if error.retry_after is not None:
                headers = {"Retry-After": str(int(error.retry_after.total_seconds()))}
            raise _http_error(
                429,
                ApiErrorCode.WORKFLOW_BACKPRESSURE,
                "workflow capacity is temporarily unavailable",
                retryable=True,
                headers=headers,
            ) from error
        except RemoteWorkflowFailureError as error:
            raise _http_error(
                503 if error.retryable else 422,
                error.code,
                error.safe_message,
                retryable=error.retryable,
            ) from error
        except UnknownDispatchOutcomeError as error:
            raise _http_error(
                503,
                ApiErrorCode.WORKFLOW_DISPATCH_UNAVAILABLE,
                "workflow dispatch outcome is unknown",
                retryable=True,
            ) from error

    @app.post(
        "/v1/workflows/{workflow_id}/schedule",
        response_model=ScheduleResult,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            200: {"model": ScheduleResult},
            404: {"model": SchedulerErrorResponse},
            409: {"model": SchedulerErrorResponse},
            422: {"model": SchedulerErrorResponse},
            503: {"model": SchedulerErrorResponse},
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
        responses={404: {"model": SchedulerErrorResponse}},
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
        responses={404: {"model": SchedulerErrorResponse}},
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


async def _run_while_connected[T](
    request: Request,
    operation: Awaitable[T],
) -> T:
    operation_task = asyncio.ensure_future(operation)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _pending = await asyncio.wait(
            {operation_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return await operation_task
        raise asyncio.CancelledError
    finally:
        for task in (operation_task, disconnect_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(operation_task, disconnect_task, return_exceptions=True)


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


def _http_error(
    status_code: int,
    code: ApiErrorCode | str,
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
