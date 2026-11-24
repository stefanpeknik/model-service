"""Run an analysis while the caller waits, without a durable execution record."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from model_service.catalog.analyses import resolve_run
from model_service.contracts.analysis_http import AnalysisRunRequest
from model_service.contracts.failures import InvalidInvocation
from model_service.control_plane.analysis_requests import check_invocation


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.catalog import Catalog
    from model_service.contracts.result import RunResult
    from model_service.control_plane.routes.v1.analyses import ImmediateAnalysisRequest
    from model_service.control_plane.scheduler.protocols import AnalysisDispatch


class SynchronousAnalysisService:
    """Resolves an analysis at admission and runs it on the runner while the caller waits."""

    def __init__(
        self, catalog: Callable[[], Catalog], dispatch: AnalysisDispatch
    ) -> None:
        """Keep where the deployed analyses come from and the dispatch to the runner.

        Args:
            catalog: Gives the catalog in use (it may be reloaded between runs).
            dispatch: The HTTP client to the analysis runner.
        """
        self._catalog = catalog
        self._dispatch = dispatch

    async def run(
        self, analysis_id: str, request: ImmediateAnalysisRequest
    ) -> RunResult:
        """Resolve the analysis with the request's settings, check the request, run it.

        The runner gets the resolved document, so it executes what was
        checked here even if the catalog changes meanwhile. The invocation's
        `configuration` records every setting the run uses.

        Args:
            analysis_id: The analysis to run.
            request: Inputs, settings (`configuration`), output path and deadline.

        Returns:
            Each output's file and media type.

        Raises:
            UnknownAnalysisError: If no deployed analysis has this ID.
            InvalidInvocation: For an invalid setting, unknown or missing
                inputs, a deadline that has passed, or `output_path` for an
                analysis with several outputs.
            ConfirmedRejectionError: If the runner refused for lack of capacity.
            RemoteFailureError: If the run failed, e.g. `model_contract_mismatch`.
            UnknownOutcomeError: If it is unknown whether the run happened.
        """
        resolved = resolve_run(self._catalog(), analysis_id, request.configuration)
        analysis = resolved.analysis
        invocation = replace(
            request.invocation(request.deadline_at),
            configuration=analysis.settings(),
        )
        if invocation.deadline_at <= datetime.now(UTC):
            raise InvalidInvocation("deadline_at must be in the future")
        check_invocation(analysis, invocation)
        return await self._dispatch.run_analysis(
            AnalysisRunRequest(invocation, resolved.document)
        )
