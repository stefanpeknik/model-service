"""Expose scheduling, polling, cancellation, immediate workflows, and raw inference through FastAPI.

Routes translate application errors into safe HTTP responses. Durable readiness checks
storage; workflow deployment health checks Ray independently.
"""

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
from model_service.health import WorkflowDeploymentState, WorkflowDeploymentStatus
from model_service.inference import create_inference_router
from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    JsonValue,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.dispatch import (
    ConfirmedDispatchRejectionError,
    RemoteWorkflowFailureError,
    UnknownDispatchOutcomeError,
)
from workflow_execution.http import WorkflowErrorResponse


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from model_service.inference import InferenceProxy


class ApiErrorCode(StrEnum):
    """Name safe public API failures for admission, execution lookup, and immediate dispatch."""

    INVALID_SCHEDULE_REQUEST = "invalid_schedule_request"
    INVALID_WORKFLOW_INVOCATION = "invalid_workflow_invocation"
    UNKNOWN_WORKFLOW = "unknown_workflow"
    SCHEDULE_CONFLICT = "schedule_conflict"
    SCHEDULING_UNAVAILABLE = "scheduling_unavailable"
    UNKNOWN_EXECUTION = "unknown_execution"
    WORKFLOW_BACKPRESSURE = "workflow_backpressure"
    WORKFLOW_DISPATCH_UNAVAILABLE = "workflow_dispatch_unavailable"


@dataclass(frozen=True, slots=True)
class ScheduleBody:
    """Accept a provider-specific schedule payload, optional timeout, and caller replay key.

    Attributes:
        provider: Integration name selecting identity, preparation, and terminal
            reporting behavior.
        body: Provider-specific JSON payload submitted for preparation.
        requested_timeout_seconds: Optional positive caller execution budget in seconds.
        idempotency_key: Caller replay key; required for standalone admission.
    """

    provider: str
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class EmpaiaScheduleBody:
    """Accept an external EMPAIA job and credential for provider-specific admission.

    Attributes:
        installation_id: Installation namespace used to scope EMPAIA job replay
            identity.
        api_url: External AppService origin.
        job_id: External EMPAIA job identifier.
        credential: Private bearer credential received for provider preparation and
            protected persistence.
        requested_timeout_seconds: Optional caller execution budget in seconds.
    """

    installation_id: str
    api_url: str
    job_id: str
    credential: str
    requested_timeout_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class SchedulerErrorDetail:
    """Describe a public API failure with its safe message and retry permission.

    Attributes:
        code: Stable safe failure identifier.
        message: Safe user-facing failure explanation.
        retryable: Whether this classified failure permits another attempt.
    """

    code: ApiErrorCode
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class SchedulerErrorResponse:
    """Wrap scheduler API error details in the standard HTTP response envelope.

    Attributes:
        detail: Structured safe HTTP error information.
    """

    detail: SchedulerErrorDetail


class ExecutionSchedulerInterface(Protocol):
    """Define the durable operations required by HTTP routes without coupling them to persistence."""

    async def check_ready(self) -> None:
        """Check durable storage availability independently of Ray workflow deployment.

        Returns:
            None: No return value.

        Raises:
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
        """
        ...

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Admit or replay a request and return whether a new execution was created.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.

        Returns:
            ScheduleResult: Public execution view and insertion-versus-replay flag.

        Raises:
            InvalidScheduleRequestError: Provider payload, configuration, paths, or
                effective deadline are invalid.
            ScheduleConflictError: A provider submission key already exists with
                different request contents.
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
            UnknownWorkflowError: The workflow is not declared in the configured
                catalog.
        """
        ...

    async def get(self, execution_id: str) -> ExecutionView:
        """Return the caller-visible execution view for an existing ID.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            ExecutionView: Caller-visible execution status and its permitted terminal
                result or error.

        Raises:
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        ...

    async def cancel(self, execution_id: str) -> ExecutionView:
        """Record cancellation and return the updated caller-visible execution view.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            ExecutionView: Caller-visible execution status and its permitted terminal
                result or error.

        Raises:
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        ...


class SynchronousWorkflowInterface(Protocol):
    """Define immediate execution that returns a result without durable admission."""

    async def run(
        self,
        workflow_id: str,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult:
        """Validate and dispatch an immediate invocation for the selected workflow.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.
        """
        ...


class DeploymentHealth(Protocol):
    """Define the separate check for Ray workflow route availability."""

    async def check(self) -> WorkflowDeploymentStatus:
        """Report compute reachability and any declared workflows missing from Ray Serve.

        Returns:
            WorkflowDeploymentStatus: Ray route availability and any missing workflow
                IDs.
        """
        ...


def create_execution_app(
    scheduler: ExecutionSchedulerInterface,
    *,
    workflow_runner: SynchronousWorkflowInterface | None = None,
    inference_proxy: InferenceProxy | None = None,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealth | None = None,
) -> FastAPI:
    """Build HTTP routes over injected services and an optional caller-owned lifespan. No resources or worker tasks start here.

    Args:
        scheduler: Durable scheduler boundary or its live HTTP client, according to the
            declared type.
        workflow_runner: Optional immediate execution service that returns results
            without durable admission.
        inference_proxy: Optional raw binary model forwarding service.
        lifespan: Optional FastAPI resource-lifetime context owned by the caller.
        deployment_health: Optional Ray workflow-route checker, independent of durable
            readiness.

    Returns:
        FastAPI: Application with registered routes and the supplied resource lifespan.
    """
    app = FastAPI(title="model-service execution scheduler", lifespan=lifespan)

    async def submit(
        request: ScheduleRequest,
        response: Response,
    ) -> ScheduleResult:
        """Translate admission errors and set 202 for a new execution or 200 for an identical replay.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.
            response: Upstream HTTP response or mutable FastAPI response used to select
                the public status.

        Returns:
            ScheduleResult: Public execution view and insertion-versus-replay flag.

        Raises:
            HTTPException: Admission reports an unknown workflow, conflicting replay,
                invalid request, or unavailable scheduling; the route returns 404, 409,
                422, or 503 respectively.
        """
        try:
            result = await scheduler.schedule(request)
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

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        """Return safe validation messages without reflecting request bodies or credentials.

        Args:
            request: Incoming public request whose path selects the safe invocation or
                admission error code.
            _error: Unused safe domain error or injected integration exception to
                classify; retained to match the interface.

        Returns:
            JSONResponse: Safe JSON validation or availability response.
        """
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
        """Dispatch immediate work while the client remains connected and translate classified failures into HTTP responses.

        Args:
            request: Incoming HTTP request monitored for caller disconnection.
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.

        Raises:
            HTTPException: The runner is unavailable, the workflow/invocation is
                invalid, dispatch is rejected, or the remote outcome fails/is unknown.
                Status and retry metadata preserve that classification.
            asyncio.CancelledError: The HTTP caller disconnects or the owning request
                task is cancelled.
        """
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
        """Translate the generic schedule body into a provider request and submit it for durable admission.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            body: Provider name/body, replay key, and optional caller timeout from
                public HTTP.
            response: Upstream HTTP response or mutable FastAPI response used to select
                the public status.

        Returns:
            ScheduleResult: Public execution view and insertion-versus-replay flag.

        Raises:
            HTTPException: Admission fails; submit translates the domain failure into
                its public HTTP status.
        """
        return await submit(
            ScheduleRequest(
                workflow_id=workflow_id,
                provider=body.provider,
                body=body.body,
                requested_timeout_seconds=body.requested_timeout_seconds,
                idempotency_key=body.idempotency_key,
            ),
            response,
        )

    @app.post(
        "/v1/providers/empaia/workflows/{workflow_id}/schedule",
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
    async def schedule_empaia(
        workflow_id: str,
        body: EmpaiaScheduleBody,
        response: Response,
    ) -> ScheduleResult:
        """Translate EMPAIA job fields into an EMPAIA provider request for durable admission.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            body: Installation, external job origin/ID, private credential, and optional
                timeout.
            response: Upstream HTTP response or mutable FastAPI response used to select
                the public status.

        Returns:
            ScheduleResult: Public execution view and insertion-versus-replay flag.

        Raises:
            HTTPException: EMPAIA admission fails; submit translates the domain failure
                into its public HTTP status.
        """
        return await submit(
            ScheduleRequest(
                workflow_id=workflow_id,
                provider="empaia",
                body={
                    "installation_id": body.installation_id,
                    "api_url": body.api_url,
                    "job_id": body.job_id,
                    "credential": body.credential,
                },
                requested_timeout_seconds=body.requested_timeout_seconds,
            ),
            response,
        )

    @app.get(
        "/v1/executions/{execution_id}",
        response_model=ExecutionView,
        responses={
            404: {"model": SchedulerErrorResponse},
            503: {"model": SchedulerErrorResponse},
        },
    )
    async def get_execution(execution_id: str) -> ExecutionView:
        """Read a public execution view, mapping absent IDs to 404 and storage outages to 503.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            ExecutionView: Caller-visible execution status and its permitted terminal
                result or error.

        Raises:
            HTTPException: Durable storage is unavailable or the execution is absent,
                producing HTTP 503 or 404.
        """
        try:
            return await scheduler.get(execution_id)
        except SchedulingUnavailableError as error:
            raise _http_error(
                503, ApiErrorCode.SCHEDULING_UNAVAILABLE, str(error), retryable=True
            ) from error
        except UnknownExecutionError as error:
            raise _http_error(
                404,
                ApiErrorCode.UNKNOWN_EXECUTION,
                "execution does not exist",
            ) from error

    @app.post(
        "/v1/executions/{execution_id}/cancel",
        response_model=ExecutionView,
        responses={
            404: {"model": SchedulerErrorResponse},
            503: {"model": SchedulerErrorResponse},
        },
    )
    async def cancel_execution(execution_id: str) -> ExecutionView:
        """Cancel a durable execution, mapping absent IDs and storage outages to safe HTTP errors.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            ExecutionView: Caller-visible execution status and its permitted terminal
                result or error.

        Raises:
            HTTPException: Durable storage is unavailable or the execution is absent,
                producing HTTP 503 or 404.
        """
        try:
            return await scheduler.cancel(execution_id)
        except SchedulingUnavailableError as error:
            raise _http_error(
                503, ApiErrorCode.SCHEDULING_UNAVAILABLE, str(error), retryable=True
            ) from error
        except UnknownExecutionError as error:
            raise _http_error(
                404,
                ApiErrorCode.UNKNOWN_EXECUTION,
                "execution does not exist",
            ) from error

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        """Report that this HTTP process is responsive without checking external services.

        Returns:
            dict[str, str]: Liveness response with status set to ok.
        """
        return {"status": "ok"}

    @app.get("/readyz")
    async def readiness() -> dict[str, object]:
        """Check repository availability so durable API readiness does not depend on Ray.

        Returns:
            dict[str, object]: Readiness response with status set to ok after durable
                storage responds.

        Raises:
            HTTPException: Durable storage is unavailable, producing HTTP 503.
        """
        try:
            await scheduler.check_ready()
        except SchedulingUnavailableError as error:
            raise _http_error(
                503,
                ApiErrorCode.SCHEDULING_UNAVAILABLE,
                "execution repository is temporarily unavailable",
                retryable=True,
            ) from error
        return {"status": "ok"}

    @app.get("/healthz/workflows")
    async def workflow_health(response: Response) -> WorkflowDeploymentStatus:
        """Report workflow deployment state separately, using 503 for configured compute that is not ready.

        Args:
            response: Upstream HTTP response or mutable FastAPI response used to select
                the public status.

        Returns:
            WorkflowDeploymentStatus: Ray route availability and any missing workflow
                IDs.
        """
        if deployment_health is None:
            return WorkflowDeploymentStatus(WorkflowDeploymentState.NOT_CONFIGURED)
        health = await deployment_health.check()
        if health.status is not WorkflowDeploymentState.READY:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return health

    app.include_router(create_inference_router(inference_proxy))

    return app


async def _run_while_connected[T](
    request: Request,
    operation: Awaitable[T],
) -> T:
    """Race an immediate operation against client disconnect, cancelling and joining both tasks on exit.

    Args:
        request: Incoming HTTP request monitored while workflow dispatch runs.
        operation: Immediate async work raced against the caller's disconnect.

    Returns:
        T: Completed immediate operation result while the caller remains connected.

    Raises:
        asyncio.CancelledError: The owning task, caller, deadline, or lease monitor
            cancels the operation.
    """
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
    """Consume ASGI receive messages until the client disconnects.

    Args:
        request: Incoming HTTP request whose ASGI events reveal disconnection.

    Returns:
        None: No return value.
    """
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
    """Construct the shared safe error envelope with an HTTP status, retry flag, and optional response headers.

    Args:
        status_code: HTTP status to return or classify.
        code: Stable failure identifier, or HTTP status when classifying an upstream
            response.
        message: Safe user-facing failure or cancellation text.
        retryable: Whether another attempt is permitted for this classified failure.
        headers: HTTP metadata to filter, send, or include in an error response.

    Returns:
        HTTPException: Structured safe HTTP error ready to raise at the route boundary.
    """
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "retryable": retryable},
        headers=headers,
    )
