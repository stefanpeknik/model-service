"""`/v1/empaia`: the remote execution protocol of EMPAIA's Job Execution Service.

An EMPAIA app registered as `http::<target>/builders/<builder>/models/<model>`
reaches `POST /builders/<builder>/models/<model>` with what an app container
would get: the App API URL, the job ID and the job token. The registration names
what runs; the job's inputs and configuration are read from the App API like a
container would. Submission is idempotent per job: JES's `Idempotency-Key` is the
job ID, so a different key is rejected. The response's `Location` is then polled
with `GET` and cancelled with `DELETE`. Remote statuses are `ExecutionStatus`
values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Header, Response, status
from pydantic import Field

from model_service.facade.auth import Access, Action
from model_service.facade.errors import ERRORS, ApiErrorCode, http_error, unavailable
from model_service.facade.routes.common import submit
from model_service.providers.empaia.gateway import PROVIDER_NAME
from model_service.providers.empaia.provider import EmpaiaSchedulePayload
from model_service.scheduler.contracts import (
    ExecutionStatus,
    ScheduleRequest,
    SchedulingUnavailableError,
    UnknownExecutionError,
)


if TYPE_CHECKING:
    from model_service.facade.ports import Capacity, Scheduler
    from model_service.scheduler.contracts import ExecutionView


@dataclass(frozen=True, slots=True)
class EmpaiaExecutionRequest:
    app_api: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]
    access_token: Annotated[str, Field(min_length=1)]
    # Operator-defined container env; model-service has no use for it.
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmpaiaExecutionStatus:
    status: ExecutionStatus
    message: str | None = None


def create_router(
    scheduler: Scheduler, capacity: Capacity, access: Access | None = None
) -> APIRouter:
    """Build JES's remote execution routes under `/v1/empaia`.

    Args:
        scheduler: Admits, reads and cancels executions.
        capacity: Explains a polled execution's status.
        access: Guards the routes; everyone is let in by default.

    Returns:
        The router.
    """
    access = access or Access()
    router = APIRouter(prefix="/v1/empaia")

    @router.post(
        "/builders/{builder_id}/models/{model_id}",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": EmpaiaExecutionStatus}, **ERRORS},
        dependencies=[access.require(Action.SUBMIT_PROVIDER_JOB)],
    )
    async def execute(
        builder_id: str,
        model_id: str,
        body: EmpaiaExecutionRequest,
        response: Response,
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> EmpaiaExecutionStatus:
        if idempotency_key is not None and idempotency_key != body.job_id:
            raise http_error(
                422,
                ApiErrorCode.INVALID_REQUEST,
                "Idempotency-Key must be the job ID",
            )
        result = await submit(
            scheduler,
            ScheduleRequest(
                builder_id=builder_id,
                model_id=model_id,
                provider=PROVIDER_NAME,
                body=EmpaiaSchedulePayload(
                    # Job IDs are unique per installation; its App API URL names it.
                    installation_id=body.app_api.rstrip("/"),
                    api_url=body.app_api,
                    job_id=body.job_id,
                    credential=body.access_token,
                ).model_dump(mode="json", exclude={"mode"}),
            ),
            response,
        )
        response.headers["Location"] = f"/v1/empaia/executions/{result.execution.id}"
        return remote_status(result.execution)

    @router.get(
        "/executions/{execution_id}",
        responses=ERRORS,
        dependencies=[access.require(Action.READ_EXECUTION)],
    )
    async def get_status(execution_id: str) -> EmpaiaExecutionStatus:
        try:
            execution = await scheduler.get(execution_id)
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error
        except UnknownExecutionError as error:
            raise _unknown_execution() from error
        return remote_status(await capacity.explain(execution))

    @router.delete(
        "/executions/{execution_id}",
        responses=ERRORS,
        dependencies=[access.require(Action.CANCEL_EXECUTION)],
    )
    async def cancel(execution_id: str) -> EmpaiaExecutionStatus:
        try:
            return remote_status(await scheduler.cancel(execution_id))
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error
        except UnknownExecutionError as error:
            raise _unknown_execution() from error

    return router


def remote_status(execution: ExecutionView) -> EmpaiaExecutionStatus:
    """Translate an execution into JES's status; a failure or cancellation is final only once EMPAIA has been told.

    JES ends the job as soon as it sees a final status; if the App Service had
    not received the failure report yet, the job would end without its reason.

    Args:
        execution: The execution.

    Returns:
        `RUNNING` while provider finish is pending; otherwise the execution's
        status, with its error message if it failed.
    """
    if execution.provider_pending:
        return EmpaiaExecutionStatus(ExecutionStatus.RUNNING)
    if execution.status is ExecutionStatus.FAILED:
        message = execution.error.message if execution.error is not None else None
        return EmpaiaExecutionStatus(ExecutionStatus.FAILED, message)
    return EmpaiaExecutionStatus(execution.status, execution.message)


def _unknown_execution() -> Exception:
    """Build the 404 error for an execution that does not exist.

    Returns:
        The HTTP error to raise.
    """
    return http_error(404, ApiErrorCode.UNKNOWN_EXECUTION, "execution does not exist")
