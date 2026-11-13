"""The client for `spatial-scores` models: per-tile score maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from model_service.contracts.model_declarations import (
    InferenceContract,
    SpatialScoresModel,
)
from model_service.deployed_models.model_clients.base import ModelClient


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class SpatialScoresClient(ModelClient[SpatialScoresModel]):
    """Client for `spatial-scores` models: one score map per tile."""

    contract = InferenceContract.SPATIAL_SCORES
    methods = ("get_config", "predict")

    def _matches(self, running: dict[str, object]) -> bool:
        """Compare the running geometry with the declaration.

        Other keys the model reports are ignored.

        Args:
            running: At least `tile_size`, `output_tile_size`, `n_channels` and
                `mpp`.

        Returns:
            Whether every one is valid and equals the declaration's.
        """
        keys = self.declaration.model_dump(exclude={"id", "contract"}).keys()
        try:
            model = SpatialScoresModel.model_validate(
                {"id": self.declaration.id}
                | {key: running[key] for key in keys if key in running}
            )
        except ValidationError:
            return False
        return model == self.declaration

    async def predict(self, tile: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Score one tile.

        Args:
            tile: CHW uint8 tile of the declared `tile_size`.

        The model may answer flat, as `(n_channels, output_tile_size,
        output_tile_size)`, or as `(output_tile_size, output_tile_size)` for one
        channel.

        Returns:
            Finite scores shaped `(n_channels, output_tile_size, output_tile_size)`.

        Raises:
            BuilderFailure: The call failed (retryable), or returned another shape
                or a non-finite value (not retryable).
        """
        import numpy as np

        model = self.declaration
        shape = (model.n_channels, model.output_tile_size, model.output_tile_size)
        try:
            scores = np.asarray(await self._call("predict", tile), dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise self._invalid_output() from error
        accepted = {shape, (scores.size,), ()} | (
            {shape[1:]} if model.n_channels == 1 else set()
        )
        if (
            scores.size != np.prod(shape)
            or scores.shape not in accepted
            or not np.isfinite(scores).all()
        ):
            raise self._invalid_output()
        return scores.reshape(shape)
