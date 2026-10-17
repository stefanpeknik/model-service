"""`/v1/empaia`: the remote execution protocol of EMPAIA's Job Execution Service.

An EMPAIA app registered as `http::<target>/<app>` reaches `POST /apps/<app>`
with what an app container would get: the App API URL, the job ID and the job
token. The app decides the builder and model; the job's inputs, mode and
configuration are read from the App API like a container would. The response's
`Location` is then polled with `GET` and cancelled with `DELETE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Response, status
from pydantic import Field

from model_service.facade.routes.common import (
    ERRORS,
    ApiErrorCode,
    Scheduler,
    http_error,
    submit,
    unavailable,
)
from model_service.providers.empaia.apps import UnknownAppError
from model_service.scheduler.contracts import (
    ExecutionStatus,
    ScheduleRequest,
    SchedulingUnavailableError,
    UnknownExecutionError,
)


if TYPE_CHECKING:
    from model_service.providers.empaia.apps import EmpaiaApps
    from model_service.scheduler.contracts import ExecutionView


type RemoteStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class EmpaiaExecutionRequest:
    app_api: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]
    access_token: Annotated[str, Field(min_length=1)]
    # Operator-defined container env; model-service has no use for it.
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmpaiaExecutionStatus:
    status: RemoteStatus
    message: str | None = None


def create_router(scheduler: Scheduler, apps: EmpaiaApps) -> APIRouter:
    router = APIRouter(prefix="/v1/empaia")

    @router.post(
        "/apps/{app_id}",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": EmpaiaExecutionStatus}, **ERRORS},
    )
    async def execute(
        app_id: str, body: EmpaiaExecutionRequest, response: Response
    ) -> EmpaiaExecutionStatus:
        try:
            app = apps.get(app_id)
        except UnknownAppError as error:
            raise http_error(
                404, ApiErrorCode.UNKNOWN_APP, "EMPAIA app is not configured"
            ) from error
        result = await submit(
            scheduler,
            ScheduleRequest(
                builder_id=app.builder,
                model_id=app.model,
                provider="empaia",
                body={
                    # Job IDs are unique per installation; its App API URL names it.
                    "installation_id": body.app_api.rstrip("/"),
                    "api_url": body.app_api,
                    "job_id": body.job_id,
                    "credential": body.access_token,
                },
            ),
            response,
        )
        response.headers["Location"] = f"/v1/empaia/executions/{result.execution.id}"
        return remote_status(result.execution)

    @router.get("/executions/{execution_id}", responses=ERRORS)
    async def get_status(execution_id: str) -> EmpaiaExecutionStatus:
        try:
            return remote_status(await scheduler.get(execution_id))
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error
        except UnknownExecutionError as error:
            raise _unknown_execution() from error

    @router.delete("/executions/{execution_id}", responses=ERRORS)
    async def cancel(execution_id: str) -> EmpaiaExecutionStatus:
        try:
            return remote_status(await scheduler.cancel(execution_id))
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error
        except UnknownExecutionError as error:
            raise _unknown_execution() from error

    return router


def remote_status(execution: ExecutionView) -> EmpaiaExecutionStatus:
    """A failure or cancellation is final only once EMPAIA has been told.

    JES ends the job as soon as it sees a final status; if the App Service had
    not received the failure report yet, the job would end without its reason.
    """
    if execution.provider_pending:
        return EmpaiaExecutionStatus("running")
    if execution.status is ExecutionStatus.FAILED:
        message = execution.error.message if execution.error is not None else None
        return EmpaiaExecutionStatus("failed", message)
    return EmpaiaExecutionStatus(_REMOTE[execution.status])


_REMOTE: dict[ExecutionStatus, RemoteStatus] = {
    ExecutionStatus.QUEUED: "queued",
    ExecutionStatus.RUNNING: "running",
    ExecutionStatus.SUCCEEDED: "succeeded",
    ExecutionStatus.FAILED: "failed",
    ExecutionStatus.CANCELLED: "cancelled",
}


def _unknown_execution() -> Exception:
    return http_error(404, ApiErrorCode.UNKNOWN_EXECUTION, "execution does not exist")
