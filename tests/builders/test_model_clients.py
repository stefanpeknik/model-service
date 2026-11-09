"""Each contract's model client verifies the running model and checks its answers."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from model_service.builders.contracts import (
    BuilderFailure,
    FailureCode,
    InferenceContract,
    SpatialScoresModel,
    TileEmbeddingModel,
)
from model_service.builders.model_clients import (
    CLIENTS,
    SpatialScoresClient,
    TileEmbeddingClient,
    client_for,
)
from tests.support.builder_fixtures import FakeDeployment, running_config


SPATIAL = SpatialScoresModel(
    id="classifier", tile_size=4, output_tile_size=2, n_channels=1, mpp=0.5
)
EMBEDDING = TileEmbeddingModel(
    id="encoder", tile_size=4, mpp=0.5, summaries=("global", "mean")
)
TILE = np.zeros((3, 4, 4), dtype=np.uint8)


def spatial_client(config: dict[str, object], score: object = None) -> Any:
    deployment = FakeDeployment(config, predict=lambda _tile: score)
    return SpatialScoresClient(SPATIAL, lambda _id: deployment)  # type: ignore[arg-type, return-value]


def embedding_client(config: dict[str, object], features: object = None) -> Any:
    deployment = FakeDeployment(config, embed=lambda _tile: features)
    return TileEmbeddingClient(EMBEDDING, lambda _id: deployment)  # type: ignore[arg-type, return-value]


class ClientRegistryTests(unittest.TestCase):
    def test_every_contract_has_a_client(self) -> None:
        self.assertEqual(set(CLIENTS), set(InferenceContract))
        self.assertIsInstance(client_for(SPATIAL), SpatialScoresClient)
        self.assertIsInstance(client_for(EMBEDDING), TileEmbeddingClient)


class VerifyTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_model_running_as_declared_passes(self) -> None:
        await spatial_client(running_config(SPATIAL)).verify()
        await embedding_client(running_config(EMBEDDING)).verify()

    async def test_a_running_model_that_differs_is_rejected(self) -> None:
        valid = running_config(SPATIAL)
        cases = [
            spatial_client(valid | {"tile_size": 8}),
            spatial_client(valid | {"mpp": 0.25}),
            spatial_client(valid | {"n_channels": 2}),
            spatial_client(valid | {"tile_size": True}),
            spatial_client(valid | {"mpp": float("nan")}),
            spatial_client({"tile_size": 4}),
            spatial_client(valid | {"unexpected": 1}),
            embedding_client(running_config(EMBEDDING) | {"tile_size": 8}),
            embedding_client(running_config(EMBEDDING) | {"summaries": ["global"]}),
        ]
        for client in cases:
            with (
                self.subTest(client=client),
                self.assertRaises(BuilderFailure) as raised,
            ):
                await client.verify()
            self.assertEqual(raised.exception.code, FailureCode.MODEL_CONTRACT_MISMATCH)
            self.assertFalse(raised.exception.retryable)

    async def test_the_handle_is_looked_up_once(self) -> None:
        lookups: list[str] = []
        deployment = FakeDeployment(running_config(SPATIAL))

        def lookup(model_id: str) -> Any:
            lookups.append(model_id)
            return deployment

        client = SpatialScoresClient(SPATIAL, lookup)
        self.assertEqual(lookups, [])
        await client.verify()
        await client.verify()
        self.assertEqual(lookups, ["classifier"])


class CallTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_are_shaped_by_the_declaration(self) -> None:
        client = spatial_client(running_config(SPATIAL), [0.1, 0.2, 0.3, 0.4])

        scores = await client.predict(TILE)

        self.assertEqual(scores.shape, (1, 2, 2))
        self.assertEqual(scores.dtype, np.float32)

    async def test_features_are_returned_as_given(self) -> None:
        features = np.ones((2, 8), dtype=np.float32)
        client = embedding_client(running_config(EMBEDDING), features)

        self.assertIs(await client.embed(TILE), features)

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
            (
                TileEmbeddingClient(
                    EMBEDDING,
                    lambda _id: FakeDeployment({}, embed=fail),  # type: ignore[arg-type, return-value]
                ),
                "embed",
            ),
        ):
            with self.subTest(call), self.assertRaises(BuilderFailure) as raised:
                await getattr(client, call)(TILE)
            self.assertEqual(raised.exception.code, FailureCode.MODEL_FAILED)
            self.assertTrue(raised.exception.retryable)
            self.assertNotIn("CUDA", raised.exception.safe_message)

    async def test_an_answer_outside_the_contract_is_a_permanent_failure(self) -> None:
        config = running_config(EMBEDDING)
        cases = [
            (spatial_client(running_config(SPATIAL), [0.5]), "predict"),
            (
                spatial_client(running_config(SPATIAL), [0.1, 0.2, np.nan, 0.4]),
                "predict",
            ),
            (spatial_client(running_config(SPATIAL), "high"), "predict"),
            (embedding_client(config, np.ones(8, dtype=np.float32)), "embed"),
            (embedding_client(config, np.ones((2, 8), dtype=np.float64)), "embed"),
            (embedding_client(config, np.ones((2, 0), dtype=np.float32)), "embed"),
            (
                embedding_client(config, np.full((2, 8), np.inf, dtype=np.float32)),
                "embed",
            ),
            (embedding_client(config, [[1.0]]), "embed"),
        ]
        for client, call in cases:
            with self.subTest(call), self.assertRaises(BuilderFailure) as raised:
                await getattr(client, call)(TILE)
            self.assertEqual(raised.exception.code, FailureCode.INVALID_MODEL_OUTPUT)
            self.assertFalse(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()
