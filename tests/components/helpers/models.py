"""A fake model for `model.` dimensions and model calls in tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FakeInput:
    """What the fake model takes.

    Attributes:
        tile_size: Tile edge in pixels.
        mpp: Microns per pixel.
    """

    tile_size: int = 224
    mpp: float = 0.5


@dataclass(frozen=True)
class FakeOutput:
    """What the fake model returns.

    Attributes:
        kind: `scores` (channels x size x size per tile) or `embedding`.
        channels: Score channels.
        size: Score map edge.
        dim: Embedding length.
        summaries: Embedding summary names; one vector each per tile.
    """

    kind: str = "scores"
    channels: int = 1
    size: int = 1
    dim: int = 8
    summaries: tuple[str, ...] = ("tile",)


@dataclass(frozen=True)
class FakeModel:
    """A model stand-in.

    Attributes:
        id: The model ID.
        input: What it takes.
        output: What it returns.
    """

    id: str = "fake-model"
    input: FakeInput = field(default_factory=FakeInput)
    output: FakeOutput = field(default_factory=FakeOutput)

    def __call__(self, tiles: np.ndarray) -> np.ndarray:
        """Score tiles by their mean brightness, so outputs follow inputs.

        Args:
            tiles: uint8 tiles, (batch, 3, tile_size, tile_size).

        Returns:
            float32 scores (batch, channels, size, size) in [0, 1], or
            embeddings (batch, dim) for one summary, else (batch, summaries,
            dim), each summary scaled by its position.
        """
        mean = tiles.reshape(len(tiles), -1).mean(axis=1).astype(np.float32) / 255
        out = self.output
        if out.kind == "embedding":
            vectors = np.repeat(mean[:, None], out.dim, axis=1) * np.linspace(
                0, 1, out.dim, dtype=np.float32
            )
            if len(out.summaries) == 1:
                return vectors
            return np.stack(
                [vectors * (i + 1) for i in range(len(out.summaries))], axis=1
            )
        shape = (len(tiles), out.channels, out.size, out.size)
        return np.broadcast_to(mean[:, None, None, None], shape).astype(np.float32)

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        """Infer one tile, as a model host's handle does.

        Args:
            tile: uint8 `(3, tile_size, tile_size)`.

        Returns:
            Scores `(channels, size, size)`, or an embedding `(dim,)` for one
            summary, else `(summaries, dim)`.
        """
        return self(tile[None])[0]


def fake_model(output: dict[str, Any] | None = None, **input_: Any) -> FakeModel:
    """Build a fake model.

    Args:
        output: Output fields (`kind`, `channels`, `size`, `dim`,
            `summaries`), as in a model file's `output:`.
        **input_: Input fields (`tile_size`, `mpp`).

    Returns:
        The fake model.
    """
    return FakeModel(input=FakeInput(**input_), output=FakeOutput(**(output or {})))
