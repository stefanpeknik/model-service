"""`/v1/analyses`: the deployed analyses, and synchronous runs of one.

`POST /v1/analyses/{id}/run` takes the builder run's request (inputs, their
access, settings as `configuration`, output path, deadline) without a model:
the analysis names its served models. The API resolves the analysis with the
settings and sends the resolved document to the analysis runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from model_service.api.http.auth import Access, Action
from model_service.api.http.errors import ERRORS
from model_service.api.http.synchronous import run_synchronously, runs_unavailable
from model_service.api.routes.v1.builders import ImmediateBuilderRequest
from model_service.catalog.analyses import FOLDER
from model_service.contracts.json_values import JsonValue
from model_service.contracts.result import BuilderResult, OutputKind
from model_service.formats.known import get_format


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.api.synchronous_runs import SynchronousAnalysisService
    from model_service.catalog import Catalog
    from model_service.catalog.analyses import Analysis


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
    access: Access,
) -> APIRouter:
    """Build the `/v1/analyses` router: the catalog and synchronous runs.

    Args:
        catalog: Gives the catalog in use.
        runner: Runs analyses synchronously; None makes `/run` answer 503.
        access: Guards the routes.

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
        request: Request, analysis_id: str, body: ImmediateBuilderRequest
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

    return router
