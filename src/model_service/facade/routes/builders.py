"""`/v1/builders`: the builder catalog, synchronous runs and standalone scheduling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fastapi import APIRouter, Request, Response, status

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
    UnknownBuilderError,
    UnknownModelError,
)
from model_service.builders.registry import BUILDERS
from model_service.facade.requests import BuilderRequest, ImmediateBuilderRequest
from model_service.facade.routes.common import (
    ERRORS,
    ApiErrorCode,
    Scheduler,
    http_error,
    lookup_error,
    submit,
)
from model_service.scheduler.contracts import ScheduleRequest, ScheduleResult


if TYPE_CHECKING:
    from collections.abc import Awaitable


@dataclass(frozen=True, slots=True)
class ScheduleBody:
    """`body` is provider-specific; for `standalone` it is a builder request."""

    provider: str
    body: dict[str, JsonValue]
    requested_timeout_seconds: int | None = None
    idempotency_key: str | None = None


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


class BuilderRunner(Protocol):
    async def run(
        self, builder_id: str, request: ImmediateBuilderRequest
    ) -> BuilderResult: ...


def create_router(
    scheduler: Scheduler, builder_runner: BuilderRunner | None
) -> APIRouter:
    router = APIRouter(prefix="/v1/builders")

    @router.get("")
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

    @router.post("/{builder_id}/run", responses=ERRORS)
    async def run_builder(
        request: Request, builder_id: str, body: ImmediateBuilderRequest
    ) -> BuilderResult:
        if builder_runner is None:
            raise http_error(
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
            raise lookup_error(error) from error
        except InvalidInvocation as error:
            raise http_error(422, ApiErrorCode.INVALID_REQUEST, str(error)) from error
        except ConfirmedRejectionError as error:
            headers = None
            if error.retry_after is not None:
                headers = {"Retry-After": str(int(error.retry_after.total_seconds()))}
            raise http_error(
                429,
                ApiErrorCode.BUILDER_BACKPRESSURE,
                "builder capacity is temporarily unavailable",
                retryable=True,
                headers=headers,
            ) from error
        except RemoteFailureError as error:
            raise http_error(
                503 if error.retryable else 422,
                error.code,
                error.safe_message,
                retryable=error.retryable,
            ) from error
        except UnknownOutcomeError as error:
            raise http_error(
                503,
                ApiErrorCode.BUILDER_UNAVAILABLE,
                "builder outcome is unknown",
                retryable=True,
            ) from error

    @router.post(
        "/{builder_id}/schedule",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": ScheduleResult}, **ERRORS},
    )
    async def schedule_builder(
        builder_id: str, body: ScheduleBody, response: Response
    ) -> ScheduleResult:
        model_id = body.body.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise http_error(
                422, ApiErrorCode.INVALID_REQUEST, "body.model_id is required"
            )
        if body.provider == "standalone":
            # Validate the shape here so errors are reported before admission.
            try:
                BuilderRequest.model_validate(body.body)
            except ValueError as error:
                raise http_error(
                    422, ApiErrorCode.INVALID_REQUEST, "request body is invalid"
                ) from error
        return await submit(
            scheduler,
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

    return router


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


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return
