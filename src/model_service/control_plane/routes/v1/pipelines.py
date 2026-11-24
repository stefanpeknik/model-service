"""`/v1/pipelines`: the deployed pipelines, and synchronous and scheduled runs of one.

`POST /v1/pipelines/{id}/run` takes the run's request (inputs, their access,
settings as `configuration`, output path, deadline); the pipeline names its
models. The control plane resolves the pipeline with the
settings and sends the resolved document to the pipeline runner.

`POST /v1/pipelines/{id}/schedule` takes the same request with a timeout
instead of a deadline, and the `Idempotency-Key` header. Admission resolves
the pipeline and stores the document on the execution; every attempt sends
that document. `/v1/executions` reads and cancels it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import Field

from model_service.contracts.caller_request import RunRequest
from model_service.contracts.invocation import Deadline
from model_service.contracts.json_values import JsonValue
from model_service.contracts.result import OutputKind, RunResult
from model_service.control_plane.http.auth import Access, Action
from model_service.control_plane.http.errors import ERRORS
from model_service.control_plane.http.scheduling import require_idempotency_key, submit
from model_service.control_plane.http.synchronous import (
    run_synchronously,
    runs_unavailable,
)
from model_service.control_plane.scheduler.public import (
    PipelineScheduleRequest,
    ScheduleResult,
)
from model_service.formats.known import get_format


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.catalog import Catalog
    from model_service.catalog.pipelines import Pipeline
    from model_service.control_plane.scheduler.protocols import Scheduler
    from model_service.control_plane.scheduler.public import ProviderName
    from model_service.control_plane.synchronous_runs import SynchronousPipelineService


class ImmediatePipelineRequest(RunRequest):
    """A synchronous pipeline run: the request plus a deadline.

    Attributes:
        deadline_at: When the run must have finished.
    """

    deadline_at: Deadline


class ScheduledPipelineRequest(RunRequest):
    """A request plus a timeout; the `Idempotency-Key` header names it."""

    requested_timeout_seconds: Annotated[int, Field(strict=True, gt=0)] | None = None

    def stored_body(self) -> dict[str, JsonValue]:
        """Serialize the request without the scheduling timeout.

        Returns:
            The JSON body stored with the execution.
        """
        return self.model_dump(mode="json", exclude={"requested_timeout_seconds"})


@dataclass(frozen=True, slots=True)
class PipelineInputView:
    """A pipeline input, as the catalog lists it.

    Attributes:
        required: Whether a run must give it.
    """

    required: bool


@dataclass(frozen=True, slots=True)
class PipelineOutputView:
    """A pipeline output, as the catalog lists it.

    Attributes:
        kind: How it is returned: always a file on shared storage.
        format: The data format's name, e.g. `score_grid`.
        media_type: The format's versioned media type.
    """

    kind: OutputKind
    format: str
    media_type: str


@dataclass(frozen=True, slots=True)
class PipelineView:
    """A pipeline, as the catalog lists it.

    Attributes:
        id: The pipeline ID, also its route.
        summary: What it does.
        inputs: What the components read, by name.
        outputs: What the sinks write, by name.
        settings: The per-run settings with the file's values.
        settings_schema: The JSON Schema of the `configuration` a run may send.
        models: The models it calls, in step order.
        providers: Where else it is offered (the file's `providers`); empty for none.
    """

    id: str
    summary: str
    inputs: dict[str, PipelineInputView]
    outputs: dict[str, PipelineOutputView]
    settings: dict[str, JsonValue]
    settings_schema: dict[str, JsonValue]
    models: list[str]
    providers: dict[str, JsonValue]


def pipeline_view(pipeline: Pipeline) -> PipelineView:
    """Describe a pipeline for the public catalog.

    Args:
        pipeline: The resolved pipeline.

    Returns:
        Its inputs, outputs with formats, settings, models and providers.
    """
    outputs = {}
    for sink in pipeline.sinks:
        found = get_format(sink.format)
        outputs[sink.options.output] = PipelineOutputView(
            OutputKind.SHARED_ARTIFACT, found.name, found.media_type
        )
    return PipelineView(
        id=pipeline.id,
        summary=pipeline.summary,
        inputs={
            name: PipelineInputView(required)
            for name, required in pipeline.inputs.items()
        },
        outputs=outputs,
        settings=pipeline.settings(),
        settings_schema=pipeline.settings_schema(),
        models=list(pipeline.models),
        providers=deepcopy(pipeline.declaration.get("providers") or {}),
    )


def create_router(
    catalog: Callable[[], Catalog],
    runner: SynchronousPipelineService | None,
    scheduler: Scheduler,
    access: Access,
    provider: ProviderName,
) -> APIRouter:
    """Build the `/v1/pipelines` router: the catalog, synchronous and scheduled runs.

    Args:
        catalog: Gives the catalog in use.
        runner: Runs pipelines synchronously; None makes `/run` answer 503.
        scheduler: Admits durable executions.
        access: Guards the routes.
        provider: The provider whose files scheduled executions read and write.

    Returns:
        The router.
    """
    router = APIRouter(prefix="/v1/pipelines")

    @router.get("", dependencies=[access.require(Action.READ_CATALOG)])
    async def list_pipelines() -> list[PipelineView]:
        """List the deployed pipelines.

        Returns:
            Every pipeline with its inputs, outputs, settings and models.
        """
        return [pipeline_view(pipeline) for pipeline in catalog().pipelines.values()]

    @router.post(
        "/{pipeline_id}/run",
        responses=ERRORS,
        dependencies=[access.require(Action.RUN)],
    )
    async def run_pipeline(
        request: Request, pipeline_id: str, body: ImmediatePipelineRequest
    ) -> RunResult:
        """Run a pipeline and wait for its result.

        The run is abandoned when the client disconnects.

        Args:
            request: The HTTP request, watched for disconnects.
            pipeline_id: The pipeline to run.
            body: Inputs, settings (`configuration`), output path and deadline.

        Returns:
            Each output's file and media type.

        Raises:
            HTTPException: Synchronous runs are unavailable, the pipeline is
                unknown, the request invalid, or the run failed (e.g.
                `model_contract_mismatch` when a model runs another
                declaration than the one resolved here).
        """
        if runner is None:
            raise runs_unavailable("synchronous pipeline runs")
        return await run_synchronously(request, runner.run(pipeline_id, body))

    @router.post(
        "/{pipeline_id}/schedule",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": ScheduleResult}, **ERRORS},
        dependencies=[access.require(Action.SCHEDULE)],
    )
    async def schedule_pipeline(
        pipeline_id: str,
        body: ScheduledPipelineRequest,
        response: Response,
        idempotency_key: Annotated[str, Depends(require_idempotency_key)],
    ) -> ScheduleResult:
        """Schedule a pipeline run as a durable execution.

        Args:
            pipeline_id: The pipeline to run.
            body: Inputs, settings (`configuration`), output path and
                requested timeout.
            response: The response, 202 for a new execution or 200 for a
                repeat.
            idempotency_key: The caller's key; repeating it returns the same
                execution.

        Returns:
            The execution's ID and status.
        """
        return await submit(
            scheduler,
            PipelineScheduleRequest(
                pipeline_id=pipeline_id,
                provider=provider,
                body=body.stored_body(),
                requested_timeout_seconds=body.requested_timeout_seconds,
                idempotency_key=idempotency_key,
            ),
            response,
        )

    return router
