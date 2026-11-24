"""The served-model host core: one resolved served model, ready to infer.

Per request, in order:

1. the tile is checked against `input:` (uint8 `(3, tile_size, tile_size)`);
2. `pre:` steps run on the tile alone, as a one-row batch with column `tile`;
3. tiles are batched (`batching.max_size`, `batching.wait_s`) and the runtime
   runs on the stacked batch;
4. `post:` steps run on the tile's output alone, as a one-row batch with
   column `output`, so only reduced data leaves the replica;
5. the output is checked against `output:` (shape, dtype, finite, range).

A raw request for unpooled tokens skips steps 4 and 5; the raw route then
drops the register tokens itself (`replies.unpooled`).

Steps run one tile at a time, so a tile's result never depends on the batch
it arrived in.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from components import RuntimeContext
from model_service.served_model.batching import Batcher
from model_service.served_model.guards import check_output, check_rows, check_tile
from model_service.served_model.replies import raw_reply


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from components import Step
    from model_service.catalog.served_models import ServedModel
    from model_service.served_model.replies import RawReply

logger = logging.getLogger(__name__)


class ServedModelHost:
    """Serves one served model: batching, guards, the raw reply and `describe()`.

    It needs an asyncio event loop for `infer` and `raw`, but no Serve
    replica, so tests and tools use it directly.

    Attributes:
        model: The served model.
    """

    def __init__(self, model: ServedModel, *, cache_root: Path) -> None:
        """Fetch the weights and load the runtime.

        Args:
            model: The resolved served model.
            cache_root: Where runtime caches go, under
                `<cache_root>/<id>/<first 12 hex digits of the digest>`.
        """
        self.model = model
        weights = model.loader.fetch(Path(tempfile.mkdtemp(prefix="weights-")))
        self._run = model.runtime.load(
            weights,
            RuntimeContext(
                cache_dir=cache_root / model.id / model.digest[:12],
                tile_size=model.input.tile_size,
                max_batch_size=model.batching.max_size,
            ),
        )
        self._batcher: Batcher[np.ndarray, np.ndarray] = Batcher(
            self._run_batch,
            max_size=model.batching.max_size,
            wait_s=model.batching.wait_s,
        )
        described = self.describe()
        logger.info(
            "served model %s built: digest %s, weights %s, execution providers %s",
            model.id,
            described["digest"],
            described["weights_revision"],
            described["execution_providers"] or "not reported",
        )

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        """Infer one tile, batched with concurrent calls.

        Args:
            tile: uint8 RGB `(3, tile_size, tile_size)`.

        Returns:
            Scores `(channels, size, size)`, or the embedding: `(dim,)` for one
            summary, else `(len(summaries), dim)`.

        Raises:
            ModelInputError: For a tile that is not what `input:` declares.
            ModelOutputError: For an output that breaks `output:`.
        """
        return await self._infer(tile, True)

    async def raw(self, body: bytes, headers: Mapping[str, str]) -> RawReply:
        """Answer a raw-inference request in the reply format of this output.

        Args:
            body: The LZ4-compressed uint8 tile, `(tile_size, tile_size, 3)`.
            headers: The request headers.

        Returns:
            The reply.
        """
        return await raw_reply(self.model, body, headers, self._infer)

    def describe(self) -> dict[str, Any]:
        """What is running, for comparison with the catalog.

        Returns:
            `id`, `digest` (the catalog computes the same one),
            `weights_revision` and `execution_providers` (the ones actually
            in use, so a fallback from TensorRT is visible).
        """
        return {
            "id": self.model.id,
            "digest": self.model.digest,
            "weights_revision": self.model.loader.revision(),
            "execution_providers": self.model.runtime.active_providers(),
        }

    async def _infer(self, tile: np.ndarray, post: bool) -> np.ndarray:
        """Infer one tile; without `post`, return the runtime's output unchecked.

        Args:
            tile: uint8 RGB `(3, tile_size, tile_size)`.
            post: Whether to run the post steps and check the output.

        Returns:
            The tile's output.
        """
        check_tile(self.model, tile)
        output = await self._batcher.submit(_apply(self.model.pre, "tile", tile))
        if not post:
            return output
        return check_output(self.model, _apply(self.model.post, "output", output))

    def _run_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """Run the runtime on one batch and split it back into tiles.

        Args:
            tiles: The batch's tiles, after the pre steps.

        Returns:
            One output per tile.
        """
        output = np.asarray(self._run(np.stack(tiles, axis=0)))
        check_rows(self.model, output, len(tiles))
        return list(output)


def _apply(steps: Sequence[Step], column: str, value: np.ndarray) -> np.ndarray:
    """Run steps on one item as a one-row batch.

    Args:
        steps: The steps, in order.
        column: The column they read and write.
        value: The item.

    Returns:
        The item after the steps.
    """
    for step in steps:
        value = step({column: value[None]})[column][0]
    return value
