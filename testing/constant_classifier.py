"""A `spatial-scores` model that returns a constant score.

It needs no GPU or model artifact, so builders, scheduling and deployment can
be exercised cheaply. Its output says nothing about real model behaviour.
"""

from __future__ import annotations

from typing import TypedDict

import lz4.frame
import numpy as np
from fastapi import FastAPI, Request
from ray import serve


class Config(TypedDict, total=False):
    """The deployment's user_config; every key is optional.

    Attributes:
        tile_size: Side of the square input tile in pixels.
        output_tile_size: Side of the score map per tile.
        n_channels: Score channels per tile.
        mpp: Microns per pixel the model reads at.
        score: The score returned for every tile.
    """

    tile_size: int
    output_tile_size: int
    n_channels: int
    mpp: float
    score: float


fastapi = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi)
class ConstantClassifier:
    """Serves a constant score for every tile, as a `spatial-scores` model."""

    def __init__(self) -> None:
        """Start with the default configuration until Ray Serve passes one."""
        self.reconfigure({})

    def reconfigure(self, config: Config) -> None:
        """Apply a new user_config.

        Args:
            config: The deployment's user_config.
        """
        self.tile_size = config.get("tile_size", 128)
        self.output_tile_size = config.get("output_tile_size", 1)
        self.n_channels = config.get("n_channels", 1)
        self.mpp = config.get("mpp", 0.5)
        self.score = config.get("score", 0.75)

    def get_config(self) -> dict[str, float | int]:
        """Report the geometry the model runs with.

        Returns:
            `tile_size`, `output_tile_size`, `n_channels` and `mpp`.
        """
        return {
            "tile_size": self.tile_size,
            "output_tile_size": self.output_tile_size,
            "n_channels": self.n_channels,
            "mpp": self.mpp,
        }

    def predict(self, image: np.ndarray) -> list[float]:
        """Score one tile.

        Args:
            image: CHW uint8 tile of `tile_size`.

        Returns:
            The constant score, once per output cell and channel.

        Raises:
            ValueError: The tile has another shape.
        """
        if image.shape != (3, self.tile_size, self.tile_size):
            raise ValueError("unexpected tile shape")
        return [self.score] * (self.n_channels * self.output_tile_size**2)

    @fastapi.post("/")
    async def classify(self, request: Request) -> float:
        """Score one LZ4-compressed HWC tile posted to the route.

        Args:
            request: The request whose body is the compressed tile.

        Returns:
            The tile's first score.
        """
        image = np.frombuffer(
            lz4.frame.decompress(await request.body()), dtype=np.uint8
        ).reshape(self.tile_size, self.tile_size, 3)
        return self.predict(image.transpose(2, 0, 1))[0]


app = ConstantClassifier.bind()  # type: ignore[attr-defined]
