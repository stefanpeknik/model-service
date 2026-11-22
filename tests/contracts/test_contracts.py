"""Check the wire format of invocations, results and analysis run requests."""

import unittest
from datetime import UTC, datetime

from pydantic import TypeAdapter, ValidationError

from model_service.contracts.analysis_http import ANALYSIS_RUN_REQUEST
from model_service.contracts.invocation import INVOCATION, ExecutionContext, InputKind
from model_service.contracts.result import RESULT, OutputKind


INVOCATION_BODY = {
    "inputs": {
        "slide": {"kind": "shared_path", "path": "slide.svs"},
        "region_of_interest": {"kind": "provider", "name": "region_of_interest"},
    },
    "configuration": None,
    "input_access": {"kind": "shared_storage", "state": {"root": "/mnt/data"}},
    "deadline_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    "output_path": None,
}


class ContractTests(unittest.TestCase):
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

    def test_run_request_nests_invocation_analysis_and_context(self) -> None:
        body = {
            "invocation": INVOCATION_BODY,
            "analysis": {"id": "prostate-heatmap", "steps": []},
            "execution_context": {
                "processing_attempt_id": "attempt-1",
                "execution_id": "execution-1",
                "protected_context": {"credential": "secret"},
            },
        }
        request = ANALYSIS_RUN_REQUEST.validate_python(body)

        self.assertNotIn("secret", repr(request.execution_context))
        self.assertEqual(ANALYSIS_RUN_REQUEST.dump_python(request, mode="json"), body)
        with self.assertRaises(ValidationError):
            ANALYSIS_RUN_REQUEST.validate_python({"invocation": INVOCATION_BODY})

    def test_execution_context_rejects_invalid_attempts_and_non_json_secrets(
        self,
    ) -> None:
        for attempt in ("", True, 1):
            with self.subTest(attempt=attempt), self.assertRaises(ValidationError):
                ExecutionContext(processing_attempt_id=attempt)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            ExecutionContext(protected_context={"invalid": float("nan")})
        with self.assertRaises(ValidationError):
            TypeAdapter(ExecutionContext).validate_python({"unexpected": True})


if __name__ == "__main__":
    unittest.main()
