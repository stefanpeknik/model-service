"""Run a builder or an analysis while the caller waits, without a durable execution record."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from model_service.api.analysis_requests import check_invocation
from model_service.builders.registry import get_builder
from model_service.catalog.analyses import resolve_run
from model_service.contracts.analysis_http import AnalysisRunRequest
from model_service.contracts.failures import InvalidInvocation
from model_service.contracts.invocation import ExecutionContext


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.api.model_catalog import ModelCatalogView
    from model_service.api.routes.v1.builders import ImmediateBuilderRequest
    from model_service.api.scheduler.protocols import AnalysisDispatch, BuilderDispatch
    from model_service.catalog import Catalog
    from model_service.contracts.result import BuilderResult


class SynchronousBuilderService:
    """Runs a builder with a model while the caller waits."""

    def __init__(self, catalog: ModelCatalogView, dispatch: BuilderDispatch) -> None:
        """Keep the catalog runs are checked against and the dispatch to builders.

        Args:
            catalog: The model catalog.
            dispatch: The HTTP client to builder replicas.
        """
        self._catalog = catalog
        self._dispatch = dispatch

    async def run(
        self, builder_id: str, model_id: str, request: ImmediateBuilderRequest
    ) -> BuilderResult:
        """Validate the run and dispatch it, waiting for the builder's result.

        Args:
            builder_id: The builder to run.
            model_id: The model the builder uses.
            request: Inputs, configuration and deadline.

        Returns:
            The builder's result.

        Raises:
            UnknownBuilderError: If the builder is not registered.
            UnknownModelError: If the model is not in the catalog.
            IncompatibleModelError: If the builder does not accept the model's contract.
            InvalidInvocation: If the deadline or inputs are invalid.
            ConfirmedRejectionError: If the builder refused for lack of capacity.
            RemoteFailureError: If the builder failed.
            UnknownOutcomeError: If it is unknown whether the builder ran.
        """
        spec = get_builder(builder_id)
        model = self._catalog.get(model_id)
        spec.check_model(model)
        invocation = request.invocation(request.deadline_at)
        spec.check_invocation(invocation)
        return await self._dispatch.run(
            builder_id, invocation, model, ExecutionContext()
        )


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
        self, analysis_id: str, request: ImmediateBuilderRequest
    ) -> BuilderResult:
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
