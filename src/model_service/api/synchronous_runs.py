"""Run a builder while the caller waits, without a durable execution record."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.registry import get_builder
from model_service.contracts.invocation import ExecutionContext


if TYPE_CHECKING:
    from model_service.api.routes.builders import ImmediateBuilderRequest
    from model_service.contracts.result import BuilderResult
    from model_service.deployed_models.catalog import ModelCatalog
    from model_service.scheduler.builder_dispatch import HTTPBuilderDispatch


class SynchronousBuilderService:
    """Runs a builder with a model while the caller waits."""

    def __init__(self, catalog: ModelCatalog, dispatch: HTTPBuilderDispatch) -> None:
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
