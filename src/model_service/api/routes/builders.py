"""`/v1/builders`: the builder catalog, and runs of a builder with a model.

`/v1/builders/{builder}/models/{model}/run` waits for the result;
`.../schedule` admits a durable execution with shared-storage inputs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import Field

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
from model_service.http_common.scheduling import require_idempotency_key, submit
from model_service.providers.standalone.provider import STANDALONE_PROVIDER
from model_service.scheduler.builder_dispatch import (
    ConfirmedRejectionError,
    RemoteFailureError,
    UnknownOutcomeError,
)
from model_service.scheduler.public import ScheduleRequest, ScheduleResult


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from model_service.api.synchronous_runs import SynchronousBuilderService
    from model_service.scheduler.protocols import Scheduler


class ImmediateBuilderRequest(BuilderRequest):
    """A synchronous builder run: the builder request plus a deadline.

    Attributes:
        deadline_at: When the run must have finished.
    """

    deadline_at: Deadline


class ScheduledBuilderRequest(BuilderRequest):
    """A builder request plus a timeout; the `Idempotency-Key` header names it."""

    requested_timeout_seconds: Annotated[int, Field(strict=True, gt=0)] | None = None

    def builder_request(self) -> dict[str, JsonValue]:
        """Serialize the builder request without the scheduling timeout.

        Returns:
            The JSON body stored with the execution.
        """
        return self.model_dump(mode="json", exclude={"requested_timeout_seconds"})


@dataclass(frozen=True, slots=True)
class BuilderInputView:
    """A builder input, as the catalog lists it.

    Attributes:
        required: Whether an invocation must provide it.
        media_type: The expected media type, if declared.
    """

    required: bool
    media_type: str | None


@dataclass(frozen=True, slots=True)
class BuilderOutputView:
    """A builder output, as the catalog lists it.

    Attributes:
        kind: How the output is returned (inline or as a shared artifact).
        media_type: The output's media type, if declared.
    """

    kind: str
    media_type: str | None


@dataclass(frozen=True, slots=True)
class BuilderView:
    """A builder, as the catalog lists it.

    Attributes:
        id: The builder's id.
        accepts: The model contract it accepts.
        inputs: Its inputs by name.
        outputs: Its outputs by name.
        configuration: Its configuration defaults.
    """

    id: str
    accepts: InferenceContract
    inputs: dict[str, BuilderInputView]
    outputs: dict[str, BuilderOutputView]
    configuration: dict[str, JsonValue]


def create_router(
    scheduler: Scheduler,
    builder_runner: SynchronousBuilderService | None,
    access: Access,
) -> APIRouter:
    """Build the `/v1/builders` router: the catalog, synchronous runs and scheduling.

    Args:
        scheduler: Admits durable executions.
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
        """List the registered builders.

        Returns:
            Every builder with its inputs, outputs and configuration defaults.
        """
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
        """Run a builder with a model and wait for its result.

        The run is abandoned when the client disconnects.

        Args:
            request: The HTTP request, watched for disconnects.
            builder_id: The builder to run.
            model_id: The model it uses.
            body: Inputs, configuration and deadline.

        Returns:
            The builder's result.

        Raises:
            HTTPException: Synchronous runs are unavailable, the builder or model
                is unknown or incompatible, or the builder failed.
        """
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
        dependencies=[access.require(Action.SCHEDULE)],
    )
    async def schedule_builder(
        builder_id: str,
        model_id: str,
        body: ScheduledBuilderRequest,
        response: Response,
        idempotency_key: Annotated[str, Depends(require_idempotency_key)],
    ) -> ScheduleResult:
        """Schedule a builder run with a model as a durable execution.

        Args:
            builder_id: The builder to run.
            model_id: The model it uses.
            body: Inputs, configuration and requested timeout.
            response: The response, 201 for a new execution or 200 for a repeat.
            idempotency_key: The caller's key; repeating it returns the same
                execution.

        Returns:
            The execution's id and status.
        """
        return await submit(
            scheduler,
            ScheduleRequest(
                builder_id=builder_id,
                model_id=model_id,
                provider=STANDALONE_PROVIDER,
                body=body.builder_request(),
                requested_timeout_seconds=body.requested_timeout_seconds,
                idempotency_key=idempotency_key,
            ),
            response,
        )

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
