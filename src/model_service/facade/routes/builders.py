"""`/v1/builders`: the builder catalog, and runs of a builder with a model.

`/v1/builders/{builder}/models/{model}/run` waits for the result;
`.../schedule` admits a durable execution with shared-storage inputs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Protocol

from fastapi import APIRouter, Depends, Request, Response, status

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

# FastAPI reads these annotations at runtime.
from model_service.facade.requests import (
    ImmediateBuilderRequest,  # noqa: TC001
    ScheduledBuilderRequest,  # noqa: TC001
)
from model_service.facade.routes.common import (
    ERRORS,
    ApiErrorCode,
    Scheduler,
    http_error,
    lookup_error,
    require_idempotency_key,
    submit,
)
from model_service.scheduler.contracts import (
    ProviderName,
    ScheduleRequest,
    ScheduleResult,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable


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
        self, builder_id: str, model_id: str, request: ImmediateBuilderRequest
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

    @router.post("/{builder_id}/models/{model_id}/run", responses=ERRORS)
    async def run_builder(
        request: Request,
        builder_id: str,
        model_id: str,
        body: ImmediateBuilderRequest,
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
                request, builder_runner.run(builder_id, model_id, body)
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
        "/{builder_id}/models/{model_id}/schedule",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": ScheduleResult}, **ERRORS},
    )
    async def schedule_builder(
        builder_id: str,
        model_id: str,
        body: ScheduledBuilderRequest,
        response: Response,
        idempotency_key: Annotated[str, Depends(require_idempotency_key)],
    ) -> ScheduleResult:
        return await submit(
            scheduler,
            ScheduleRequest(
                builder_id=builder_id,
                model_id=model_id,
                provider=ProviderName.STANDALONE,
                body=body.builder_request(),
                requested_timeout_seconds=body.requested_timeout_seconds,
                idempotency_key=idempotency_key,
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
