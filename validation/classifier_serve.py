"""Host a deterministic classifier for native processor plumbing tests.

The model uses 128-pixel inputs and a constant score. Successful checks establish
transport and geometry behavior rather than real model accuracy.
"""

from __future__ import annotations

import lz4.frame
import numpy as np
from fastapi import FastAPI, Request
from ray import serve


fastapi = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi)
class ValidationProstateClassifier:
    """Return deterministic model metadata and scores through both Ray handles and raw HTTP."""

    def get_config(self) -> dict[str, float | int]:
        """Declare 128-pixel input tiles, one scalar output, and a target resolution of 0.5 micrometers per pixel.

        Returns:
            dict[str, float | int]: Input tile size 128, output extent one, one score
                channel, and target mpp 0.5.
        """
        return {
            "tile_size": 128,
            "output_tile_size": 1,
            "n_channels": 1,
            "mpp": 0.5,
        }

    def predict(self, image: np.ndarray) -> list[float]:
        """Require a 3-by-128-by-128 channel-first tile and return the fixed score 0.75.

        Args:
            image: Channel-first uint8 image with shape 3 by 128 by 128 in the
                validation classifier.

        Returns:
            list[float]: One scalar score, fixed at 0.75 for the deterministic
                validation model.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if image.shape != (3, 128, 128):
            raise ValueError("unexpected validation tile shape")
        return [0.75]

    @fastapi.post("/")
    async def classify(self, request: Request) -> float:
        """Decode an LZ4-compressed HWC uint8 tile, transpose it to CHW, and return the constant scalar prediction.

        Args:
            request: Request to validate, admit, dispatch, forward, or process, as
                specified by its declared type.

        Returns:
            float: Fixed validation score of 0.75 after validating the compressed image
                payload.
        """
        image = np.frombuffer(
            lz4.frame.decompress(await request.body()), dtype=np.uint8
        ).reshape(128, 128, 3)
        return self.predict(image.transpose(2, 0, 1))[0]


app = ValidationProstateClassifier.bind()  # type: ignore[attr-defined]
