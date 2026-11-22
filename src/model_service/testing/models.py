"""A fake served model for `model.` dimensions and model calls in tests."""

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
    """

    kind: str = "scores"
    channels: int = 1
    size: int = 1
    dim: int = 8


@dataclass(frozen=True)
class FakeModel:
    """A served-model stand-in.

    Attributes:
        id: The served-model ID.
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
            embeddings (batch, dim).
        """
        mean = tiles.reshape(len(tiles), -1).mean(axis=1).astype(np.float32) / 255
        out = self.output
        if out.kind == "embedding":
            return np.repeat(mean[:, None], out.dim, axis=1) * np.linspace(
                0, 1, out.dim, dtype=np.float32
            )
        shape = (len(tiles), out.channels, out.size, out.size)
        return np.broadcast_to(mean[:, None, None, None], shape).astype(np.float32)

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        """Infer one tile, as a served-model host's handle does.

        Args:
            tile: uint8 `(3, tile_size, tile_size)`.

        Returns:
            Scores `(channels, size, size)`, or an embedding `(dim,)`.
        """
        return self(tile[None])[0]


def fake_model(output: dict[str, Any] | None = None, **input_: Any) -> FakeModel:
    """Build a fake served model.

    Args:
        output: Output fields (`kind`, `channels`, `size`, `dim`), as in a
            served-model file's `output:`.
        **input_: Input fields (`tile_size`, `mpp`).

    Returns:
        The fake model.
    """
    return FakeModel(input=FakeInput(**input_), output=FakeOutput(**(output or {})))
