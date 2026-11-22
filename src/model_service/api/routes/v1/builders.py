"""`/v1/builders`: the builder catalog, and runs of a builder with a model.

`/v1/builders/{builder}/models/{model}/run` waits for the result;
`.../schedule` admits a durable execution with shared-storage inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import Field

from model_service.api.http.auth import Access, Action
from model_service.api.http.errors import ERRORS
from model_service.api.http.scheduling import require_idempotency_key, submit
from model_service.api.http.synchronous import run_synchronously, runs_unavailable
from model_service.api.scheduler.public import ScheduleRequest, ScheduleResult
from model_service.builders.registry import BUILDERS
from model_service.contracts.caller_request import BuilderRequest
from model_service.contracts.invocation import Deadline
from model_service.contracts.json_values import JsonValue
from model_service.contracts.model_declarations import InferenceContract
from model_service.contracts.result import BuilderResult


if TYPE_CHECKING:
    from model_service.api.scheduler.protocols import Scheduler
    from model_service.api.scheduler.public import ProviderName
    from model_service.api.synchronous_runs import SynchronousBuilderService


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
        configuration_schema: The JSON Schema of the `configuration` a request
            may send: each setting's type, limits, default and description.
    """

    id: str
    accepts: InferenceContract
    inputs: dict[str, BuilderInputView]
    outputs: dict[str, BuilderOutputView]
    configuration: dict[str, JsonValue]
    configuration_schema: dict[str, JsonValue]


def create_router(
    scheduler: Scheduler,
    builder_runner: SynchronousBuilderService | None,
    access: Access,
    provider: ProviderName,
) -> APIRouter:
    """Build the `/v1/builders` router: the catalog, synchronous runs and scheduling.

    Args:
        scheduler: Admits durable executions.
        builder_runner: Runs builders synchronously; None makes `/run` answer 503.
        access: Guards the routes.
        provider: The provider whose files scheduled executions read and write.

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
            Every builder with its inputs, outputs, configuration defaults and
            the schema of the settings a request may change.
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
                configuration_schema=spec.configuration.model_json_schema(),
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
            raise runs_unavailable("synchronous builder runs")
        return await run_synchronously(
            request, builder_runner.run(builder_id, model_id, body)
        )

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
                provider=provider,
                body=body.builder_request(),
                requested_timeout_seconds=body.requested_timeout_seconds,
                idempotency_key=idempotency_key,
            ),
            response,
        )

    return router
