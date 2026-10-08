"""Invoke classifier configuration and prediction through a Ray Serve deployment handle.

These calls use Ray rather than the public raw inference HTTP or binary compression
protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from workflows.heatmap.processor import ModelTileConfiguration


if TYPE_CHECKING:
    from ray.serve.handle import DeploymentHandle


class RayHeatmapPredictor:
    """Adapt one classifier application handle to the heatmap predictor interface."""

    def __init__(self, model_id: str) -> None:
        """Store the classifier application ID without acquiring a Ray handle yet.

        Args:
            model_id: Ray Serve classifier application identifier.
        """
        self._model_id = model_id
        self._handle: DeploymentHandle | None = None

    async def configuration(self) -> ModelTileConfiguration:
        """Read model configuration through Ray and validate the tile geometry and target resolution.

        Returns:
            ModelTileConfiguration: Validated model tile geometry, output channels, and
                target resolution.

        Raises:
            ValidationError: Classifier metadata fails the typed tile configuration
                schema.
        """
        return ModelTileConfiguration.from_values(
            await self._model().get_config.remote()
        )

    async def predict(self, tile: object) -> object:
        """Call the classifier's predict method through its Ray handle for one tile.

        Args:
            tile: One channel-first model input tile.

        Returns:
            object: Classifier scores returned by the Ray deployment handle.
        """
        return await self._model().predict.remote(tile)

    def _model(self) -> DeploymentHandle:
        """Acquire and cache the classifier application handle on first use.

        Returns:
            DeploymentHandle: Cached Ray Serve handle for the configured classifier
                application.
        """
        if self._handle is None:
            from ray import serve

            self._handle = serve.get_app_handle(self._model_id)
        return self._handle
