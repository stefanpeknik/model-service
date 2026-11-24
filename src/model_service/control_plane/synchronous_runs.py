"""Run a pipeline while the caller waits, without a durable execution record."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from model_service.catalog.pipelines import resolve_run
from model_service.contracts.failures import InvalidInvocation
from model_service.contracts.pipeline_http import PipelineRunRequest
from model_service.control_plane.pipeline_requests import check_invocation


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.catalog import Catalog
    from model_service.contracts.result import RunResult
    from model_service.control_plane.routes.v1.pipelines import ImmediatePipelineRequest
    from model_service.control_plane.scheduler.protocols import PipelineDispatch


class SynchronousPipelineService:
    """Resolves a pipeline at admission and runs it on the runner while the caller waits."""

    def __init__(
        self, catalog: Callable[[], Catalog], dispatch: PipelineDispatch
    ) -> None:
        """Keep where the deployed pipelines come from and the dispatch to the runner.

        Args:
            catalog: Gives the catalog in use (it may be reloaded between runs).
            dispatch: The HTTP client to the pipeline runner.
        """
        self._catalog = catalog
        self._dispatch = dispatch

    async def run(
        self, pipeline_id: str, request: ImmediatePipelineRequest
    ) -> RunResult:
        """Resolve the pipeline with the request's settings, check the request, run it.

        The runner gets the resolved document, so it executes what was
        checked here even if the catalog changes meanwhile. The invocation's
        `configuration` records every setting the run uses.

        Args:
            pipeline_id: The pipeline to run.
            request: Inputs, settings (`configuration`), output path and deadline.

        Returns:
            Each output's file and media type.

        Raises:
            UnknownPipelineError: If no deployed pipeline has this ID.
            InvalidInvocation: For an invalid setting, unknown or missing
                inputs, a deadline that has passed, or `output_path` for a
                pipeline with several outputs.
            ConfirmedRejectionError: If the runner refused for lack of capacity.
            RemoteFailureError: If the run failed, e.g. `model_contract_mismatch`.
            UnknownOutcomeError: If it is unknown whether the run happened.
        """
        resolved = resolve_run(self._catalog(), pipeline_id, request.configuration)
        pipeline = resolved.pipeline
        invocation = replace(
            request.invocation(request.deadline_at),
            configuration=pipeline.settings(),
        )
        if invocation.deadline_at <= datetime.now(UTC):
            raise InvalidInvocation("deadline_at must be in the future")
        check_invocation(pipeline, invocation)
        return await self._dispatch.run_pipeline(
            PipelineRunRequest(invocation, resolved.document)
        )
