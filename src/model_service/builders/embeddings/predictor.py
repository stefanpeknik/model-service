"""Call a `tile-embedding` model's `embed` method through a deployment handle.

A handle call passes the tile as a numpy array inside Ray, skipping the raw
HTTP route's LZ4 encoding and the Serve proxy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from model_service.builders.contracts import BuilderFailure, FailureCode


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from ray.serve.handle import DeploymentHandle

    from model_service.builders.contracts import TileEmbeddingModel


class EmbeddingPredictor(Protocol):
    async def verify(self) -> None:
        """Check that the running model matches its catalog declaration."""
        ...

    async def predict(self, tile: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Embed one tile.

        Args:
            tile: CHW uint8 RGB tile.

        Returns:
            A `(dim,)` vector, or `(len(summaries), dim)` for several summaries.
        """
        ...


class RayEmbeddingPredictor(EmbeddingPredictor):
    def __init__(
        self, model: TileEmbeddingModel, handle: DeploymentHandle[Any] | None = None
    ) -> None:
        """Bind the predictor to a model deployment.

        Args:
            model: Catalog declaration of the model.
            handle: Deployment handle to use; looked up by model id on first use
                when omitted.
        """
        self._model = model
        self._handle = handle

    async def verify(self) -> None:
        """Fail when the running model's tile size or summaries differ from the catalog.

        Raises:
            BuilderFailure: The running configuration differs (not retryable).
        """
        values = await self._deployment().get_config.remote()
        expected = {
            "tile_size": self._model.tile_size,
            "summaries": list(self._model.summaries),
        }
        if (
            not isinstance(values, dict)
            or {name: values.get(name) for name in expected} != expected
        ):
            raise BuilderFailure(
                code=FailureCode.MODEL_CONTRACT_MISMATCH,
                safe_message=(
                    f"model {self._model.id!r} is running with a configuration that "
                    "differs from its catalog declaration"
                ),
                retryable=False,
            )

    async def predict(self, tile: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Embed one tile through the model's `embed` method.

        Args:
            tile: CHW uint8 RGB tile.

        Returns:
            A `(dim,)` vector, or `(len(summaries), dim)` for several summaries.

        Raises:
            BuilderFailure: The model call failed (retryable) or returned invalid
                features (not retryable).
        """
        try:
            result = await self._deployment().embed.remote(tile)
        except Exception as error:
            raise BuilderFailure(
                code=FailureCode.MODEL_FAILED,
                safe_message="embedding model request failed",
                retryable=True,
            ) from error
        return checked_features(result, self._model)

    def _deployment(self) -> DeploymentHandle[Any]:
        """Return the model's deployment handle, looking it up on first use.

        Returns:
            The handle of the Ray Serve application named after the model id.
        """
        if self._handle is None:
            from ray import serve

            self._handle = serve.get_app_handle(self._model.id)
        return self._handle


def checked_features(value: object, model: TileEmbeddingModel) -> NDArray[np.float32]:
    """Check that a model returned finite float32 features of the declared shape.

    Args:
        value: What the model returned.
        model: Declaration giving the number of summaries.

    Returns:
        The value, unchanged.

    Raises:
        BuilderFailure: The value is not a finite, non-empty float32 array with
            one row per summary (or a vector for one summary).
    """
    import numpy as np

    summaries = len(model.summaries)
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.ndim != (1 if summaries == 1 else 2)
        or (summaries > 1 and value.shape[0] != summaries)
        or value.shape[-1] == 0
        or not np.isfinite(value).all()
    ):
        raise BuilderFailure(
            code=FailureCode.INVALID_MODEL_OUTPUT,
            safe_message="embedding model returned invalid features",
            retryable=False,
        )
    return value
