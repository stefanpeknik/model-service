"""`/v1/analyses`: the deployed analyses, and synchronous and scheduled runs of one.

`POST /v1/analyses/{id}/run` takes the run's request (inputs, their access,
settings as `configuration`, output path, deadline); the analysis names its
served models. The API resolves the analysis with the
settings and sends the resolved document to the analysis runner.

`POST /v1/analyses/{id}/schedule` takes the same request with a timeout
instead of a deadline, and the `Idempotency-Key` header. Admission resolves
the analysis and stores the document on the execution; every attempt sends
that document. `/v1/executions` reads and cancels it.
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
from model_service.api.scheduler.public import AnalysisScheduleRequest, ScheduleResult
from model_service.catalog.analyses import FOLDER
from model_service.contracts.caller_request import BuilderRequest
from model_service.contracts.invocation import Deadline
from model_service.contracts.json_values import JsonValue
from model_service.contracts.result import BuilderResult, OutputKind
from model_service.formats.known import get_format


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.api.scheduler.protocols import Scheduler
    from model_service.api.scheduler.public import ProviderName
    from model_service.api.synchronous_runs import SynchronousAnalysisService
    from model_service.catalog import Catalog
    from model_service.catalog.analyses import Analysis


class ImmediateAnalysisRequest(BuilderRequest):
    """A synchronous analysis run: the request plus a deadline.

    Attributes:
        deadline_at: When the run must have finished.
    """

    deadline_at: Deadline


class ScheduledAnalysisRequest(BuilderRequest):
    """A request plus a timeout; the `Idempotency-Key` header names it."""

    requested_timeout_seconds: Annotated[int, Field(strict=True, gt=0)] | None = None

    def stored_body(self) -> dict[str, JsonValue]:
        """Serialize the request without the scheduling timeout.

        Returns:
            The JSON body stored with the execution.
        """
        return self.model_dump(mode="json", exclude={"requested_timeout_seconds"})


@dataclass(frozen=True, slots=True)
class AnalysisInputView:
    """An analysis input, as the catalog lists it.

    Attributes:
        required: Whether a run must give it.
    """

    required: bool


@dataclass(frozen=True, slots=True)
class AnalysisOutputView:
    """An analysis output, as the catalog lists it.

    Attributes:
        kind: How it is returned: always a file on shared storage today.
        format: The data format's name, e.g. `score_grid`.
        media_type: The format's versioned media type.
    """

    kind: OutputKind
    format: str
    media_type: str


@dataclass(frozen=True, slots=True)
class AnalysisView:
    """An analysis, as the catalog lists it.

    Attributes:
        id: The analysis ID, also its route.
        summary: What it does.
        inputs: What the components read, by name.
        outputs: What the sinks write, by name.
        settings: The per-run settings with the file's values.
        settings_schema: The JSON Schema of the `configuration` a run may send.
        models: The served models it calls, in step order.
        exposure: Where else it is offered (the file's `expose`); empty for none.
    """

    id: str
    summary: str
    inputs: dict[str, AnalysisInputView]
    outputs: dict[str, AnalysisOutputView]
    settings: dict[str, JsonValue]
    settings_schema: dict[str, JsonValue]
    models: list[str]
    exposure: dict[str, JsonValue]


def analysis_view(analysis: Analysis) -> AnalysisView:
    """Describe an analysis for the public catalog.

    Args:
        analysis: The resolved analysis.

    Returns:
        Its inputs, outputs with formats, settings, models and exposure.
    """
    outputs = {}
    for sink in analysis.sinks:
        found = get_format(sink.format)
        outputs[sink.options.output] = AnalysisOutputView(
            OutputKind.SHARED_ARTIFACT, found.name, found.media_type
        )
    return AnalysisView(
        id=analysis.id,
        summary=analysis.summary,
        inputs={
            name: AnalysisInputView(required)
            for name, required in analysis.inputs.items()
        },
        outputs=outputs,
        settings=analysis.settings(),
        settings_schema=analysis.settings_schema(),
        models=list(analysis.models),
        exposure={},
    )


def create_router(
    catalog: Callable[[], Catalog],
    runner: SynchronousAnalysisService | None,
    scheduler: Scheduler,
    access: Access,
    provider: ProviderName,
) -> APIRouter:
    """Build the `/v1/analyses` router: the catalog, synchronous and scheduled runs.

    Args:
        catalog: Gives the catalog in use.
        runner: Runs analyses synchronously; None makes `/run` answer 503.
        scheduler: Admits durable executions.
        access: Guards the routes.
        provider: The provider whose files scheduled executions read and write.

    Returns:
        The router.
    """
    router = APIRouter(prefix="/v1/analyses")

    @router.get("", dependencies=[access.require(Action.READ_CATALOG)])
    async def list_analyses() -> list[AnalysisView]:
        """List the deployed analyses.

        Returns:
            Every analysis with its inputs, outputs, settings and models.
        """
        return [
            analysis_view(entry.value) for entry in catalog().entries(FOLDER).values()
        ]

    @router.post(
        "/{analysis_id}/run",
        responses=ERRORS,
        dependencies=[access.require(Action.RUN)],
    )
    async def run_analysis(
        request: Request, analysis_id: str, body: ImmediateAnalysisRequest
    ) -> BuilderResult:
        """Run an analysis and wait for its result.

        The run is abandoned when the client disconnects.

        Args:
            request: The HTTP request, watched for disconnects.
            analysis_id: The analysis to run.
            body: Inputs, settings (`configuration`), output path and deadline.

        Returns:
            Each output's file and media type.

        Raises:
            HTTPException: Synchronous runs are unavailable, the analysis is
                unknown, the request invalid, or the run failed (e.g.
                `model_contract_mismatch` when a served model runs another
                declaration than the one resolved here).
        """
        if runner is None:
            raise runs_unavailable("synchronous analysis runs")
        return await run_synchronously(request, runner.run(analysis_id, body))

    @router.post(
        "/{analysis_id}/schedule",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": ScheduleResult}, **ERRORS},
        dependencies=[access.require(Action.SCHEDULE)],
    )
    async def schedule_analysis(
        analysis_id: str,
        body: ScheduledAnalysisRequest,
        response: Response,
        idempotency_key: Annotated[str, Depends(require_idempotency_key)],
    ) -> ScheduleResult:
        """Schedule an analysis run as a durable execution.

        Args:
            analysis_id: The analysis to run.
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
            AnalysisScheduleRequest(
                analysis_id=analysis_id,
                provider=provider,
                body=body.stored_body(),
                requested_timeout_seconds=body.requested_timeout_seconds,
                idempotency_key=idempotency_key,
            ),
            response,
        )

    return router
