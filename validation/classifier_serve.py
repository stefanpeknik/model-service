from __future__ import annotations

import numpy as np
from ray import serve


@serve.deployment(num_replicas=1)
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


app = ValidationProstateClassifier.bind()  # type: ignore[attr-defined]
