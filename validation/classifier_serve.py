from __future__ import annotations

import lz4.frame
import numpy as np
from fastapi import FastAPI, Request
from ray import serve


fastapi = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi)
class ValidationProstateClassifier:
    def get_config(self) -> dict[str, float | int]:
        return {
            "tile_size": 128,
            "output_tile_size": 1,
            "n_channels": 1,
            "mpp": 0.5,
        }

    def predict(self, image: np.ndarray) -> list[float]:
        if image.shape != (3, 128, 128):
            raise ValueError("unexpected validation tile shape")
        return [0.75]

    @fastapi.post("/")
    async def classify(self, request: Request) -> float:
        image = np.frombuffer(
            lz4.frame.decompress(await request.body()), dtype=np.uint8
        ).reshape(128, 128, 3)
        return self.predict(image.transpose(2, 0, 1))[0]


app = ValidationProstateClassifier.bind()  # type: ignore[attr-defined]
