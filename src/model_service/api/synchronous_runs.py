"""Run a builder while the caller waits, without a durable execution record."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from model_service.builders.registry import get_builder
from model_service.wire.invocation import ExecutionContext


if TYPE_CHECKING:
    from model_service.api.request_bodies import ImmediateBuilderRequest
    from model_service.model_catalog import ModelCatalog
    from model_service.scheduler.builder_dispatch import BuilderDispatch
    from model_service.wire.result import BuilderResult


class SynchronousBuilderRunner(ABC):
    @abstractmethod
    async def run(
        self, builder_id: str, model_id: str, request: ImmediateBuilderRequest
    ) -> BuilderResult:
        """Run a builder with a model while the caller waits.

        Args:
            builder_id: The builder to run.
            model_id: The model the builder uses.
            request: Inputs, configuration and deadline.

        Returns:
            The builder's result.

        Raises:
            UnknownBuilderError: If the builder is not registered.
            UnknownModelError: If the model is not deployed.
            IncompatibleModelError: If the builder does not accept the model.
            InvalidInvocation: If the request is invalid.
            ConfirmedRejectionError: If the builder refused for lack of capacity.
            RemoteFailureError: If the builder failed.
            UnknownOutcomeError: If it is unknown whether the builder ran.
        """
        ...


class SynchronousBuilderService(SynchronousBuilderRunner):
    def __init__(self, catalog: ModelCatalog, dispatch: BuilderDispatch) -> None:
        """Keep the catalog and builder dispatch.

        Args:
            catalog: Deployed models.
            dispatch: Sends builder runs to Ray Serve.
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
        """
        spec = get_builder(builder_id)
        model = self._catalog.get(model_id)
        spec.check_model(model)
        invocation = request.invocation(request.deadline_at)
        spec.check_invocation(invocation)
        return await self._dispatch.run(
            builder_id, invocation, model, ExecutionContext()
        )
