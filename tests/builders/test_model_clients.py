"""Each contract's model client verifies the running model and checks its answers."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from model_service.builders.contracts import (
    BuilderFailure,
    FailureCode,
    SpatialScoresModel,
)
from model_service.builders.model_clients import SpatialScoresClient
from tests.support.builder_fixtures import FakeDeployment, running_config


SPATIAL = SpatialScoresModel(
    id="classifier", tile_size=4, output_tile_size=2, n_channels=1, mpp=0.5
)
TILE = np.zeros((3, 4, 4), dtype=np.uint8)


def spatial_client(config: dict[str, object], score: object = None) -> Any:
    deployment = FakeDeployment(config, predict=lambda _tile: score)
    return SpatialScoresClient(SPATIAL, lambda _id: deployment)  # type: ignore[arg-type, return-value]


class VerifyTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_model_running_as_declared_passes(self) -> None:
        await spatial_client(running_config(SPATIAL)).verify()
        # Keys outside the contract are the model's own business.
        await spatial_client(running_config(SPATIAL) | {"max_batch_size": 8}).verify()

    async def test_a_running_model_that_differs_is_rejected(self) -> None:
        valid = running_config(SPATIAL)
        cases = [
            spatial_client(valid | {"tile_size": 8}),
            spatial_client(valid | {"mpp": 0.25}),
            spatial_client(valid | {"n_channels": 2}),
            spatial_client(valid | {"tile_size": True}),
            spatial_client(valid | {"mpp": float("nan")}),
            spatial_client({"tile_size": 4}),
        ]
        for client in cases:
            with (
                self.subTest(client=client),
                self.assertRaises(BuilderFailure) as raised,
            ):
                await client.verify()
            self.assertEqual(raised.exception.code, FailureCode.MODEL_CONTRACT_MISMATCH)
            self.assertFalse(raised.exception.retryable)


class CallTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_are_shaped_by_the_declaration(self) -> None:
        client = spatial_client(running_config(SPATIAL), [0.1, 0.2, 0.3, 0.4])

        scores = await client.predict(TILE)

        self.assertEqual(scores.shape, (1, 2, 2))
        self.assertEqual(scores.dtype, np.float32)
        for answer in ([[0.1, 0.2], [0.3, 0.4]], [[[0.1, 0.2], [0.3, 0.4]]]):
            client = spatial_client(running_config(SPATIAL), answer)
            self.assertEqual((await client.predict(TILE)).shape, (1, 2, 2))

    async def test_a_failed_call_is_a_retryable_model_failure(self) -> None:
        def fail(_tile: object) -> object:
            raise RuntimeError("CUDA out of memory")

        for client, call in (
            (
                SpatialScoresClient(
                    SPATIAL,
                    lambda _id: FakeDeployment({}, predict=fail),  # type: ignore[arg-type, return-value]
                ),
                "predict",
            ),
        ):
            with self.subTest(call), self.assertRaises(BuilderFailure) as raised:
                await getattr(client, call)(TILE)
            self.assertEqual(raised.exception.code, FailureCode.MODEL_FAILED)
            self.assertTrue(raised.exception.retryable)
            self.assertNotIn("CUDA", raised.exception.safe_message)

    async def test_an_answer_outside_the_contract_is_a_permanent_failure(self) -> None:
        cases = [
            (spatial_client(running_config(SPATIAL), [0.5]), "predict"),
            (
                spatial_client(running_config(SPATIAL), [0.1, 0.2, np.nan, 0.4]),
                "predict",
            ),
            (spatial_client(running_config(SPATIAL), "high"), "predict"),
            # Right count, wrong layout: (4, 1) is not a 2x2 grid.
            (
                spatial_client(running_config(SPATIAL), [[0.1], [0.2], [0.3], [0.4]]),
                "predict",
            ),
        ]
        for client, call in cases:
            with self.subTest(call), self.assertRaises(BuilderFailure) as raised:
                await getattr(client, call)(TILE)
            self.assertEqual(raised.exception.code, FailureCode.INVALID_MODEL_OUTPUT)
            self.assertFalse(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()
