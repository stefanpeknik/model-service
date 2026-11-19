"""The client for `tile-embedding` models: feature vectors per tile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.model_clients.base import ModelClient
from model_service.contracts.model_declarations import (
    InferenceContract,
    TileEmbeddingModel,
)


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class TileEmbeddingClient(ModelClient[TileEmbeddingModel]):
    """Client for `tile-embedding` models: one vector per summary for each tile."""

    contract = InferenceContract.TILE_EMBEDDING
    methods = ("get_config", "embed")

    def _matches(self, running: dict[str, object]) -> bool:
        """Compare the running tile size and summaries with the declaration.

        Args:
            running: At least `tile_size` and `summaries`.

        Returns:
            Whether both equal the declaration's.
        """
        return running.get("tile_size") == self.declaration.tile_size and running.get(
            "summaries"
        ) == list(self.declaration.summaries)

    async def embed(self, tile: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Embed one tile.

        Args:
            tile: CHW uint8 RGB tile of the declared `tile_size`.

        Returns:
            A `(dim,)` vector for one summary, else `(len(summaries), dim)`.

        Raises:
            BuilderFailure: The call failed (retryable) or did not return finite,
                non-empty float32 features of that shape (not retryable).
        """
        import numpy as np

        features = await self._call("embed", tile)
        summaries = len(self.declaration.summaries)
        if (
            not isinstance(features, np.ndarray)
            or features.dtype != np.float32
            or features.ndim != (1 if summaries == 1 else 2)
            or (summaries > 1 and features.shape[0] != summaries)
            or features.shape[-1] == 0
            or not np.isfinite(features).all()
        ):
            raise self._invalid_output()
        return features
