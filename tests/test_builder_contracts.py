"""Check the wire format of invocations, results, run requests and model bindings."""

import unittest
from datetime import UTC, datetime

from pydantic import ValidationError

from builders.contracts import (
    INVOCATION,
    MODEL_BINDING,
    RESULT,
    InputKind,
    OutputKind,
    SpatialScoresModel,
    TileEmbeddingModel,
)
from builders.http import RUN_REQUEST


INVOCATION_BODY = {
    "inputs": {
        "slide": {"kind": "shared_path", "path": "slide.svs"},
        "region_of_interest": {"kind": "provider", "name": "region_of_interest"},
    },
    "resolved_configuration": None,
    "input_access": {"kind": "shared_storage", "state": {"root": "/mnt/data"}},
    "deadline_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    "output_path": None,
}


class BuilderContractTests(unittest.TestCase):
    def test_invocation_round_trips_with_tagged_inputs(self) -> None:
        invocation = INVOCATION.validate_python(INVOCATION_BODY)

        self.assertIs(invocation.inputs["slide"].kind, InputKind.SHARED_PATH)
        self.assertIs(invocation.inputs["region_of_interest"].kind, InputKind.PROVIDER)
        self.assertEqual(
            INVOCATION.dump_python(invocation, mode="json"), INVOCATION_BODY
        )

    def test_invocation_rejects_untagged_and_inline_inputs(self) -> None:
        for value in ({"path": "slide.svs"}, {"kind": "inline", "value": 7}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                INVOCATION.validate_python(
                    {**INVOCATION_BODY, "inputs": {"slide": value}}
                )

    def test_result_round_trips_with_tagged_outputs(self) -> None:
        body = {
            "outputs": {
                "score": {"kind": "inline", "value": 0.75},
                "heatmap": {
                    "kind": "shared_artifact",
                    "path": "/mnt/data/heatmap.tiff",
                    "media_type": "image/tiff",
                },
            }
        }
        result = RESULT.validate_python(body)

        self.assertIs(result.outputs["score"].kind, OutputKind.INLINE)
        self.assertIs(result.outputs["heatmap"].kind, OutputKind.SHARED_ARTIFACT)
        self.assertEqual(RESULT.dump_python(result, mode="json"), body)

    def test_model_binding_is_discriminated_by_contract(self) -> None:
        spatial = MODEL_BINDING.validate_python(
            {
                "id": "classifier",
                "contract": "spatial-scores",
                "tile_size": 512,
                "output_tile_size": 1,
                "n_channels": 1,
                "mpp": 0.468,
            }
        )
        embedding = MODEL_BINDING.validate_python(
            {
                "id": "virchow2",
                "contract": "tile-embedding",
                "tile_size": 224,
                "mpp": 0.5,
                "summaries": ["global", "local_mean"],
            }
        )

        self.assertIsInstance(spatial, SpatialScoresModel)
        self.assertIsInstance(embedding, TileEmbeddingModel)
        for invalid in (
            {"id": "x", "contract": "spatial-scores", "tile_size": 8, "mpp": 1},
            {
                "id": "x",
                "contract": "tile-embedding",
                "tile_size": 8,
                "mpp": 1,
                "summaries": [],
            },
            {
                "id": "x",
                "contract": "tile-embedding",
                "tile_size": "8",
                "mpp": 1,
                "summaries": ["tile"],
            },
            {"id": "x", "contract": "unknown", "tile_size": 8, "mpp": 1},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                MODEL_BINDING.validate_python(invalid)

    def test_run_request_nests_invocation_model_and_context(self) -> None:
        body = {
            "invocation": INVOCATION_BODY,
            "model": {
                "id": "virchow2",
                "contract": "tile-embedding",
                "tile_size": 224,
                "mpp": 0.5,
                "summaries": ["tile"],
            },
            "execution_context": {
                "processing_attempt_id": "attempt-1",
                "protected_context": {"credential": "secret"},
            },
        }
        request = RUN_REQUEST.validate_python(body)

        self.assertIsInstance(request.model, TileEmbeddingModel)
        self.assertNotIn("secret", repr(request.execution_context))
        self.assertEqual(RUN_REQUEST.dump_python(request, mode="json"), body)
        with self.assertRaises(ValidationError):
            RUN_REQUEST.validate_python({"invocation": INVOCATION_BODY})


if __name__ == "__main__":
    unittest.main()
