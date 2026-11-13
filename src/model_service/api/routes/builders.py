"""`/v1/builders`: the builder catalog, and runs of a builder with a model.

`/v1/builders/{builder}/models/{model}/run` waits for the result.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from model_service.builders.registry import BUILDERS
from model_service.contracts.caller_request import BuilderRequest
from model_service.contracts.failures import (
    IncompatibleModelError,
    InvalidInvocation,
    UnknownBuilderError,
    UnknownModelError,
)
from model_service.contracts.invocation import Deadline
from model_service.contracts.json_values import JsonValue
from model_service.contracts.model_declarations import InferenceContract
from model_service.contracts.result import BuilderResult
from model_service.http_common.auth import Access, Action
from model_service.http_common.errors import (
    ERRORS,
    ApiErrorCode,
    http_error,
    lookup_error,
)
from model_service.scheduler.builder_dispatch import (
    ConfirmedRejectionError,
    RemoteFailureError,
    UnknownOutcomeError,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from model_service.api.synchronous_runs import SynchronousBuilderService


class ImmediateBuilderRequest(BuilderRequest):
    deadline_at: Deadline


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


def create_router(
    builder_runner: SynchronousBuilderService | None,
    access: Access,
) -> APIRouter:
    """Build the `/v1/builders` router: the catalog and synchronous runs.

    Args:
        builder_runner: Runs builders synchronously; None makes `/run` answer 503.
        access: Guards the routes.

    Returns:
        The router.
    """
    router = APIRouter(prefix="/v1/builders")

    @router.get(
        "",
        dependencies=[access.require(Action.READ_CATALOG)],
    )
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

    @router.post(
        "/{builder_id}/models/{model_id}/run",
        responses=ERRORS,
        dependencies=[access.require(Action.RUN)],
    )
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

    return router


async def _run_while_connected[T](request: Request, operation: Awaitable[T]) -> T:
    """Await the operation, cancelling it if the client disconnects first.

    Args:
        request: The request whose connection is watched.
        operation: The work to run.

    Returns:
        The operation's result.

    Raises:
        asyncio.CancelledError: If the client disconnects before the operation ends.
    """
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
    """Return once the client disconnects.

    Args:
        request: The request whose ASGI messages are read.
    """
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return
