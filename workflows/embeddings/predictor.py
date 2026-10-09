"""Use the raw model HTTP contract so model-owned preprocessing remains unchanged."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from workflow_execution.contracts import WorkflowFailure


if TYPE_CHECKING:
    import httpx
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """Bind an embedding workflow to a raw route and its ordered output summaries.

    Attributes:
        identifier: Existing raw-model route identifier.
        summaries: Ordered semantic names for the raw output's vectors.
        tile_size: Square RGB input size expected by the raw model, in pixels.
    """

    identifier: str
    summaries: tuple[str, ...]
    tile_size: int = 224


class HttpEmbeddingPredictor:
    """Compress RGB tiles and validate the model's declared float32 feature output."""

    def __init__(self, client: httpx.AsyncClient, model: EmbeddingModel) -> None:
        """Retain a caller-owned internal HTTP client and the selected model binding.

        Args:
            client: Invocation-owned client; this adapter does not close it.
            model: Raw-model route, input geometry, and output-summary contract.
        """
        self._client = client
        self._model = model

    async def predict(self, tile: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Predict one tile using the raw model's existing preprocessing and pooling.

        Args:
            tile: Channel-first RGB uint8 pixels at the model's declared tile size.

        Returns:
            NDArray[np.float32]: Finite feature vector or ordered summary vectors.

        Raises:
            WorkflowFailure: Transport/status failure or invalid output; only transient
                transport failures and retryable HTTP statuses permit another attempt.
        """
        import httpx
        import lz4.frame
        import numpy as np

        try:
            response = await self._client.post(
                f"/{self._model.identifier}/",
                content=lz4.frame.compress(tile.transpose(1, 2, 0).tobytes()),
                headers={
                    "content-type": "application/octet-stream",
                    "x-output-dtype": "float32",
                    "x-pool-tokens": "true",
                },
            )
        except httpx.HTTPError as error:
            raise WorkflowFailure(
                code="model_unavailable",
                safe_message="embedding model is unavailable",
                retryable=True,
            ) from error
        if response.status_code != 200:
            raise WorkflowFailure(
                code="model_failed",
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
            raise WorkflowFailure(
                code="invalid_model_output",
                safe_message="embedding model returned invalid features",
                retryable=False,
            ) from error
