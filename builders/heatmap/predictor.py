"""Call a `spatial-scores` model through its Ray Serve application handle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from builders.contracts import BuilderFailure, FailureCode, SpatialScoresModel


if TYPE_CHECKING:
    from ray.serve.handle import DeploymentHandle


class RayModelPredictor:
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._handle: DeploymentHandle[Any] | None = None

    async def verify(self, declared: SpatialScoresModel) -> None:
        """Fail when the running model's geometry differs from its declaration."""
        values = await self._model().get_config.remote()
        try:
            running = SpatialScoresModel.model_validate({"id": declared.id, **values})
        except ValueError as error:
            raise BuilderFailure(
                code=FailureCode.MODEL_CONTRACT_MISMATCH,
                safe_message=f"model {declared.id!r} reports an invalid configuration",
                retryable=False,
            ) from error
        if running != declared:
            raise BuilderFailure(
                code=FailureCode.MODEL_CONTRACT_MISMATCH,
                safe_message=(
                    f"model {declared.id!r} is running with a configuration that "
                    "differs from its catalog declaration"
                ),
                retryable=False,
            )

    async def predict(self, tile: object) -> object:
        return await self._model().predict.remote(tile)

    def _model(self) -> DeploymentHandle[Any]:
        if self._handle is None:
            from ray import serve

            self._handle = serve.get_app_handle(self._model_id)
        return self._handle
