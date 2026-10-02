from __future__ import annotations

from typing import TYPE_CHECKING

from workflow_execution.heatmap import ModelTileConfiguration


if TYPE_CHECKING:
    from ray.serve.handle import DeploymentHandle


class RayHeatmapPredictor:
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._handle: DeploymentHandle | None = None

    async def configuration(self) -> ModelTileConfiguration:
        return ModelTileConfiguration.from_values(
            await self._model().get_config.remote()
        )

    async def predict(self, tile: object) -> object:
        return await self._model().predict.remote(tile)

    def _model(self) -> DeploymentHandle:
        if self._handle is None:
            from ray import serve

            self._handle = serve.get_app_handle(self._model_id)
        return self._handle
