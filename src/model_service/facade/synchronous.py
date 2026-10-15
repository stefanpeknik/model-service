"""Run a builder while the caller waits, without a durable execution record."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.application import validate_invocation, validate_model
from model_service.builders.contracts import ExecutionContext
from model_service.builders.registry import get_builder


if TYPE_CHECKING:
    from model_service.builders.client import BuilderDispatch
    from model_service.builders.contracts import BuilderResult
    from model_service.facade.catalog import ModelCatalog
    from model_service.facade.requests import ImmediateBuilderRequest


class SynchronousBuilderService:
    def __init__(self, catalog: ModelCatalog, dispatch: BuilderDispatch) -> None:
        self._catalog = catalog
        self._dispatch = dispatch

    async def run(
        self, builder_id: str, request: ImmediateBuilderRequest
    ) -> BuilderResult:
        spec = get_builder(builder_id)
        model = self._catalog.get(request.model_id)
        validate_model(spec, model)
        invocation = request.invocation(request.deadline_at)
        validate_invocation(spec, invocation)
        return await self._dispatch.run(
            builder_id, invocation, model, ExecutionContext()
        )
