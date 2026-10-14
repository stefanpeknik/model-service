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
    tile_size: int
    output_tile_size: int
    n_channels: int
    mpp: float
    score: float


fastapi = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi)
class ConstantClassifier:
    def __init__(self) -> None:
        self.reconfigure({})

    def reconfigure(self, config: Config) -> None:
        self.tile_size = config.get("tile_size", 128)
        self.output_tile_size = config.get("output_tile_size", 1)
        self.n_channels = config.get("n_channels", 1)
        self.mpp = config.get("mpp", 0.5)
        self.score = config.get("score", 0.75)

    def get_config(self) -> dict[str, float | int]:
        return {
            "tile_size": self.tile_size,
            "output_tile_size": self.output_tile_size,
            "n_channels": self.n_channels,
            "mpp": self.mpp,
        }

    def predict(self, image: np.ndarray) -> list[float]:
        if image.shape != (3, self.tile_size, self.tile_size):
            raise ValueError("unexpected tile shape")
        return [self.score] * (self.n_channels * self.output_tile_size**2)

    @fastapi.post("/")
    async def classify(self, request: Request) -> float:
        image = np.frombuffer(
            lz4.frame.decompress(await request.body()), dtype=np.uint8
        ).reshape(self.tile_size, self.tile_size, 3)
        return self.predict(image.transpose(2, 0, 1))[0]


app = ConstantClassifier.bind()  # type: ignore[attr-defined]
