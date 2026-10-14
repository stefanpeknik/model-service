"""Call a `tile-embedding` model through its raw HTTP route.

Using the raw route keeps the model's own preprocessing and token pooling.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from builders.contracts import BuilderFailure, FailureCode


if TYPE_CHECKING:
    import httpx
    import numpy as np
    from numpy.typing import NDArray

    from builders.contracts import TileEmbeddingModel


class HttpEmbeddingPredictor:
    def __init__(self, client: httpx.AsyncClient, model: TileEmbeddingModel) -> None:
        self._client = client
        self._model = model

    async def predict(self, tile: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return a `(dim,)` vector, or `(len(summaries), dim)` for several summaries."""
        import httpx
        import lz4.frame
        import numpy as np

        try:
            response = await self._client.post(
                f"/{self._model.id}/",
                content=lz4.frame.compress(tile.transpose(1, 2, 0).tobytes()),
                headers={
                    "content-type": "application/octet-stream",
                    "x-output-dtype": "float32",
                    "x-pool-tokens": "true",
                },
            )
        except httpx.HTTPError as error:
            raise BuilderFailure(
                code=FailureCode.MODEL_UNAVAILABLE,
                safe_message="embedding model is unavailable",
                retryable=True,
            ) from error
        if response.status_code != 200:
            raise BuilderFailure(
                code=FailureCode.MODEL_FAILED,
                safe_message="embedding model request failed",
                retryable=response.status_code in {408, 429}
                or response.status_code >= 500,
            )
        try:
            shape = ast.literal_eval(response.headers["x-output-shape"])
            if (
                not isinstance(shape, tuple)
                or not shape
                or any(type(size) is not int or size <= 0 for size in shape)
            ):
                raise ValueError("invalid shape")
            if len(self._model.summaries) == 1:
                if len(shape) != 1:
                    raise ValueError("expected one vector")
            elif len(shape) != 2 or shape[0] != len(self._model.summaries):
                raise ValueError("unexpected summary shape")
            result = np.frombuffer(
                lz4.frame.decompress(response.content), dtype=np.float32
            ).reshape(shape)
            if not np.isfinite(result).all():
                raise ValueError("nonfinite model output")
            return result
        except (KeyError, ValueError, SyntaxError, RuntimeError) as error:
            raise BuilderFailure(
                code=FailureCode.INVALID_MODEL_OUTPUT,
                safe_message="embedding model returned invalid features",
                retryable=False,
            ) from error
