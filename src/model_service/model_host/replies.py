"""The raw-inference route: an LZ4 tile in, a reply in the output's format out.

The format follows from `output:`, with no per-model code:

- one score per tile (`scores`, 1 channel, size 1): the score as a JSON float;
- a score map: the scores' bytes, LZ4-compressed;
- an embedding: the features' bytes, LZ4-compressed, with `x-output-shape`.
  `x-pool-tokens: false` skips `post:` and replies with the runtime's tokens
  minus the register tokens a `cls_mean_pool` post step names (`unpooled`),
  and `x-output-dtype` casts the features (float32 by default).

The request body is an LZ4-compressed uint8 RGB tile, `(tile_size,
tile_size, 3)` (channels last). Unifying these formats is a later change,
together with an SDK update.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from pydantic import TypeAdapter

from components.steps.cls_mean_pool import ClsMeanPool
from model_service.catalog.models import ScoresOutput


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from model_service.catalog.models import Model

    type Infer = Callable[[np.ndarray, bool], Awaitable[np.ndarray]]


JSON = "application/json"
BYTES = "application/octet-stream"
# FastAPI serialises a route's float return through pydantic; so does this.
FLOAT = TypeAdapter(float)


@dataclass(frozen=True)
class RawReply:
    """An HTTP reply of the raw-inference route.

    Attributes:
        content: The body.
        media_type: Its content type.
        headers: Extra headers.
    """

    content: bytes
    media_type: str
    headers: dict[str, str] = field(default_factory=dict)


async def raw_reply(
    model: Model, body: bytes, headers: Mapping[str, str], infer: Infer
) -> RawReply:
    """Answer one raw-inference request.

    Args:
        model: The model.
        body: The LZ4-compressed tile.
        headers: The request headers, lowercase names.
        infer: Infers one CHW tile, with or without `post:`.

    Returns:
        The reply.
    """
    import lz4.frame

    data = await asyncio.to_thread(lz4.frame.decompress, body)
    size = model.input.tile_size
    tile = np.frombuffer(data, dtype=np.uint8).reshape(size, size, 3)
    tile = tile.transpose(2, 0, 1)
    declared = model.output
    if isinstance(declared, ScoresOutput):
        scores = await infer(tile, True)
        if declared.channels == 1 and declared.size == 1:
            return RawReply(FLOAT.dump_json(float(scores.reshape(-1)[0])), JSON)
        return RawReply(lz4.frame.compress(scores.tobytes()), BYTES)
    post = headers.get("x-pool-tokens", "true").lower() == "true"
    dtype = np.dtype(headers.get("x-output-dtype", "float32").lower())
    output = await infer(tile, post)
    features = (output if post else unpooled(model, output)).astype(dtype, copy=False)
    return RawReply(
        lz4.frame.compress(features.tobytes()),
        BYTES,
        {"x-output-shape": str(features.shape)},
    )


def unpooled(model: Model, tokens: np.ndarray) -> np.ndarray:
    """The tokens an `x-pool-tokens: false` caller gets: no register tokens.

    Register tokens follow the class token, and a `cls_mean_pool` post step
    says how many there are (Virchow2 has 4). Pooling leaves them out, so
    unpooled callers never got them either.

    Args:
        model: The model.
        tokens: The runtime's output for one tile, `(tokens, dim)`.

    Returns:
        The class token and the patch tokens.
    """
    registers = sum(
        step.options.registers for step in model.post if isinstance(step, ClsMeanPool)
    )
    return np.delete(tokens, np.arange(1, 1 + registers), axis=0)
