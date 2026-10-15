"""Public HTTP API: builders, models, durable executions and raw inference.

Routes translate application errors into safe `{code, message, retryable}`
responses. Readiness checks durable storage; `/healthz/deployments` checks Ray.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field

from model_service.builders.client import (
    ConfirmedRejectionError,
    RemoteFailureError,
    UnknownOutcomeError,
)
from model_service.builders.contracts import (
    BuilderResult,
    IncompatibleModelError,
    InferenceContract,
    InvalidInvocation,
    JsonValue,
    ModelBinding,
    UnknownBuilderError,
    UnknownModelError,
)
from model_service.builders.registry import BUILDERS
from model_service.facade.health import DeploymentState, DeploymentStatus
from model_service.facade.inference import create_inference_router, inference_headers
from model_service.facade.requests import BuilderRequest, ImmediateBuilderRequest
from model_service.scheduler.contracts import (
    ExecutionView,
    InferenceResult,
    InferenceScheduleRequest,
    InvalidScheduleRequestError,
    ResultUnavailableError,
    ScheduleConflictError,
    ScheduleRequest,
    ScheduleResult,
    SchedulingUnavailableError,
    UnknownExecutionError,
)
from model_service.scheduler.inference import IncompatibleInputError


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from starlette.types import Lifespan

    from model_service.facade.catalog import ModelCatalog
    from model_service.facade.inference import InferenceProxy


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


MAX_LOOKUP_IDS = 1000


@dataclass(frozen=True, slots=True)
class ScheduleBody:
    """`body` is provider-specific; for `standalone` it is a builder request."""

    provider: str
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class EmpaiaScheduleBody:
    model_id: Annotated[str, Field(min_length=1)]
    installation_id: str
    api_url: str
    job_id: str
    credential: str
    requested_timeout_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionLookup:
    ids: Annotated[list[str], Field(max_length=MAX_LOOKUP_IDS)]


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    detail: ErrorDetail


@dataclass(frozen=True, slots=True)
class ModelView:
    id: str
    contract: InferenceContract
    parameters: dict[str, JsonValue]
    builders: list[str]


@dataclass(frozen=True, slots=True)
class BuilderInputView:
    required: bool
    media_type: str | None


@dataclass(frozen=True, slots=True)
class BuilderOutputView:
    kind: str
    media_type: str | None


@dataclass(frozen=True, slots=True)
class BuilderView:
    id: str
    accepts: InferenceContract
    inputs: dict[str, BuilderInputView]
    outputs: dict[str, BuilderOutputView]
    configuration: dict[str, JsonValue]


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


class BuilderRunner(Protocol):
    async def run(
        self, builder_id: str, request: ImmediateBuilderRequest
    ) -> BuilderResult: ...


class DeploymentHealth(Protocol):
    async def check(self) -> DeploymentStatus: ...


_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (404, 409, 413, 422, 429, 503)
}


def model_view(model: ModelBinding) -> ModelView:
    from model_service.facade.catalog import ModelCatalog

    return ModelView(
        id=model.id,
        contract=model.contract,
        parameters=model.model_dump(mode="json", exclude={"id", "contract"}),
        builders=ModelCatalog.compatible_builders(model),
    )


def create_app(
    scheduler: Scheduler,
    *,
    catalog: ModelCatalog,
    builder_runner: BuilderRunner | None = None,
    inference_proxy: InferenceProxy | None = None,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealth | None = None,
    inference_max_request_bytes: int = 2 * 1024 * 1024,
) -> FastAPI:
    app = FastAPI(title="model-service", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        # Do not echo the body: it may contain provider credentials.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": {
                    "code": ApiErrorCode.INVALID_REQUEST,
                    "message": "request body is invalid",
                    "retryable": False,
                }
            },
        )

    async def submit(
        request: ScheduleRequest | InferenceScheduleRequest, response: Response
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
            raise _lookup_error(error) from error
        except ScheduleConflictError as error:
            raise _http_error(
                409, ApiErrorCode.SCHEDULE_CONFLICT, str(error)
            ) from error
        except (InvalidScheduleRequestError, IncompatibleInputError) as error:
            raise _http_error(422, ApiErrorCode.INVALID_REQUEST, str(error)) from error
        except SchedulingUnavailableError as error:
            raise _http_error(
                503, ApiErrorCode.SCHEDULING_UNAVAILABLE, str(error), retryable=True
            ) from error
        response.status_code = 202 if result.created else 200
        return result

    @app.get("/v1/models")
    async def list_models() -> list[ModelView]:
        return [model_view(model) for model in catalog]

    @app.get("/v1/models/{model_id}", responses=_ERRORS)
    async def get_model(model_id: str) -> ModelView:
        try:
            return model_view(catalog.get(model_id))
        except UnknownModelError as error:
            raise _lookup_error(error) from error

    @app.post(
        "/v1/models/{model_id}/schedule",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": ScheduleResult}, **_ERRORS},
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def schedule_inference(
        model_id: str,
        request: Request,
        response: Response,
        idempotency_key: Annotated[str | None, Header(min_length=1)] = None,
        timeout_seconds: int | None = None,
    ) -> ScheduleResult:
        """Durably queue one raw model request (the body `POST /{model_id}/` takes)."""
        if idempotency_key is None:
            raise _http_error(
                422, ApiErrorCode.INVALID_REQUEST, "Idempotency-Key header is required"
            )
        payload = await _read_body(request, inference_max_request_bytes)
        return await submit(
            InferenceScheduleRequest(
                model_id=model_id,
                payload=payload,
                headers={
                    name: value
                    for name, value in inference_headers(request.headers).items()
                    if name.startswith("x-")
                },
                idempotency_key=idempotency_key,
                requested_timeout_seconds=timeout_seconds,
            ),
            response,
        )

    @app.get("/v1/builders")
    async def list_builders() -> list[BuilderView]:
        return [
            BuilderView(
                id=spec.id,
                accepts=spec.accepts,
                inputs={
                    name: BuilderInputView(item.required, item.media_type)
                    for name, item in spec.inputs.items()
                },
                outputs={
                    name: BuilderOutputView(item.kind, item.media_type)
                    for name, item in spec.outputs.items()
                },
                configuration=spec.configuration().model_dump(mode="json"),
            )
            for spec in BUILDERS.values()
        ]

    @app.post("/v1/builders/{builder_id}/run", responses=_ERRORS)
    async def run_builder(
        request: Request, builder_id: str, body: ImmediateBuilderRequest
    ) -> BuilderResult:
        if builder_runner is None:
            raise _http_error(
                503,
                ApiErrorCode.BUILDER_UNAVAILABLE,
                "synchronous builder runs are unavailable",
                retryable=True,
            )
        try:
            return await _run_while_connected(
                request, builder_runner.run(builder_id, body)
            )
        except (
            UnknownBuilderError,
            UnknownModelError,
            IncompatibleModelError,
        ) as error:
            raise _lookup_error(error) from error
        except InvalidInvocation as error:
            raise _http_error(422, ApiErrorCode.INVALID_REQUEST, str(error)) from error
        except ConfirmedRejectionError as error:
            headers = None
            if error.retry_after is not None:
                headers = {"Retry-After": str(int(error.retry_after.total_seconds()))}
            raise _http_error(
                429,
                ApiErrorCode.BUILDER_BACKPRESSURE,
                "builder capacity is temporarily unavailable",
                retryable=True,
                headers=headers,
            ) from error
        except RemoteFailureError as error:
            raise _http_error(
                503 if error.retryable else 422,
                error.code,
                error.safe_message,
                retryable=error.retryable,
            ) from error
        except UnknownOutcomeError as error:
            raise _http_error(
                503,
                ApiErrorCode.BUILDER_UNAVAILABLE,
                "builder outcome is unknown",
                retryable=True,
            ) from error

    @app.post(
        "/v1/builders/{builder_id}/schedule",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": ScheduleResult}, **_ERRORS},
    )
    async def schedule_builder(
        builder_id: str, body: ScheduleBody, response: Response
    ) -> ScheduleResult:
        model_id = body.body.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise _http_error(
                422, ApiErrorCode.INVALID_REQUEST, "body.model_id is required"
            )
        if body.provider == "standalone":
            # Validate the shape here so errors are reported before admission.
            try:
                BuilderRequest.model_validate(body.body)
            except ValueError as error:
                raise _http_error(
                    422, ApiErrorCode.INVALID_REQUEST, "request body is invalid"
                ) from error
        return await submit(
            ScheduleRequest(
                builder_id=builder_id,
                model_id=model_id,
                provider=body.provider,
                body=body.body,
                requested_timeout_seconds=body.requested_timeout_seconds,
                idempotency_key=body.idempotency_key,
            ),
            response,
        )

    @app.post(
        "/v1/providers/empaia/builders/{builder_id}/schedule",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": ScheduleResult}, **_ERRORS},
    )
    async def schedule_empaia(
        builder_id: str, body: EmpaiaScheduleBody, response: Response
    ) -> ScheduleResult:
        return await submit(
            ScheduleRequest(
                builder_id=builder_id,
                model_id=body.model_id,
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

    @app.get("/v1/executions/{execution_id}", responses=_ERRORS)
    async def get_execution(execution_id: str) -> ExecutionView:
        try:
            return await scheduler.get(execution_id)
        except SchedulingUnavailableError as error:
            raise _http_error(
                503, ApiErrorCode.SCHEDULING_UNAVAILABLE, str(error), retryable=True
            ) from error
        except UnknownExecutionError as error:
            raise _http_error(
                404, ApiErrorCode.UNKNOWN_EXECUTION, "execution does not exist"
            ) from error

    @app.post("/v1/executions/lookup", responses=_ERRORS)
    async def lookup_executions(body: ExecutionLookup) -> list[ExecutionView]:
        """Views of the listed executions that exist; unknown IDs are left out."""
        try:
            return await scheduler.get_many(body.ids)
        except SchedulingUnavailableError as error:
            raise _http_error(
                503, ApiErrorCode.SCHEDULING_UNAVAILABLE, str(error), retryable=True
            ) from error

    @app.get(
        "/v1/executions/{execution_id}/result",
        responses={
            200: {"content": {"application/octet-stream": {}}},
            **_ERRORS,
        },
    )
    async def get_inference_result(execution_id: str) -> Response:
        """The stored model response of a succeeded inference execution."""
        try:
            result, content = await scheduler.inference_result(execution_id)
        except SchedulingUnavailableError as error:
            raise _http_error(
                503, ApiErrorCode.SCHEDULING_UNAVAILABLE, str(error), retryable=True
            ) from error
        except UnknownExecutionError as error:
            raise _http_error(
                404, ApiErrorCode.UNKNOWN_EXECUTION, "execution does not exist"
            ) from error
        except ResultUnavailableError as error:
            raise _http_error(
                409,
                ApiErrorCode.RESULT_UNAVAILABLE,
                "execution has no inference response",
            ) from error
        return Response(
            content=content, media_type=result.media_type, headers=result.headers
        )

    @app.post("/v1/executions/{execution_id}/cancel", responses=_ERRORS)
    async def cancel_execution(execution_id: str) -> ExecutionView:
        try:
            return await scheduler.cancel(execution_id)
        except SchedulingUnavailableError as error:
            raise _http_error(
                503, ApiErrorCode.SCHEDULING_UNAVAILABLE, str(error), retryable=True
            ) from error
        except UnknownExecutionError as error:
            raise _http_error(
                404, ApiErrorCode.UNKNOWN_EXECUTION, "execution does not exist"
            ) from error

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readiness() -> dict[str, str]:
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

    @app.get("/healthz/deployments")
    async def deployment_status(response: Response) -> DeploymentStatus:
        if deployment_health is None:
            return DeploymentStatus(DeploymentState.NOT_CONFIGURED)
        result = await deployment_health.check()
        if result.status is not DeploymentState.READY:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    app.include_router(create_inference_router(inference_proxy))
    return app


async def _run_while_connected[T](request: Request, operation: Awaitable[T]) -> T:
    """Cancel the operation if the client disconnects first."""
    operation_task = asyncio.ensure_future(operation)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _pending = await asyncio.wait(
            {operation_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if operation_task in done:
            return await operation_task
        raise asyncio.CancelledError
    finally:
        for task in (operation_task, disconnect_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(operation_task, disconnect_task, return_exceptions=True)


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _payload_too_large(limit)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise _payload_too_large(limit)
    return bytes(body)


def _payload_too_large(limit: int) -> HTTPException:
    return _http_error(
        413, ApiErrorCode.PAYLOAD_TOO_LARGE, f"request body exceeds {limit} bytes"
    )


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


def _lookup_error(error: Exception) -> HTTPException:
    if isinstance(error, UnknownBuilderError):
        return _http_error(404, ApiErrorCode.UNKNOWN_BUILDER, "builder does not exist")
    if isinstance(error, UnknownModelError):
        return _http_error(404, ApiErrorCode.UNKNOWN_MODEL, "model is not deployed")
    return _http_error(422, ApiErrorCode.INCOMPATIBLE_MODEL, str(error))


def _http_error(
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
