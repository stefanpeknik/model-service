"""Check the wire format of invocations, results, run requests and error codes."""

import unittest
from datetime import UTC, datetime

from pydantic import TypeAdapter, ValidationError

from model_service.contracts.analysis_http import ANALYSIS_RUN_REQUEST
from model_service.contracts.analysis_http import ErrorCode as RunnerErrorCode
from model_service.contracts.caller_request import RunRequest
from model_service.contracts.failures import FailureCode
from model_service.contracts.invocation import INVOCATION, ExecutionContext, InputKind
from model_service.contracts.result import RESULT, OutputKind
from model_service.control_plane.http.errors import ApiErrorCode
from model_service.control_plane.scheduler.status import ExecutionErrorCode
from model_service.formats import embedding_table, score_grid


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
            },
            "served_models": {
                "virchow2": {
                    "digest": "d" * 64,
                    "weights_revision": "paige-ai/Virchow2@abc123",
                    "execution_providers": ["CPUExecutionProvider"],
                }
            },
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

    def test_codes_and_media_types_keep_their_wire_values(self) -> None:
        # Callers, the RationAI SDK and stored executions read these strings,
        # so renaming the Python names must never change them.
        self.assertEqual(
            {code.name: code.value for code in FailureCode},
            {
                "DEADLINE_EXCEEDED": "run_deadline_exceeded",
                "INTERNAL_ERROR": "run_internal_error",
                "INPUT_UNAVAILABLE": "input_unavailable",
                "MODEL_FAILED": "model_failed",
                "MODEL_CONTRACT_MISMATCH": "model_contract_mismatch",
                "INVALID_MODEL_OUTPUT": "invalid_model_output",
            },
        )
        self.assertEqual(
            ApiErrorCode.RUNNER_BACKPRESSURE, "analysis_runner_backpressure"
        )
        self.assertEqual(ApiErrorCode.RUNNER_UNAVAILABLE, "analysis_runner_unavailable")
        self.assertEqual(ExecutionErrorCode.RUN_OUTCOME_UNKNOWN, "run_outcome_unknown")
        self.assertEqual(RunnerErrorCode.INVALID_INVOCATION, "invalid_run_request")
        self.assertEqual(RunnerErrorCode.UNAVAILABLE, "analysis_runner_unavailable")
        self.assertEqual(
            score_grid.MEDIA_TYPE, "application/vnd.rationai.score-grid+npz; version=1"
        )
        self.assertEqual(
            embedding_table.MEDIA_TYPE,
            "application/vnd.rationai.embedding-table+hdf5; version=2",
        )

    def test_run_request_keeps_its_field_names(self) -> None:
        body = {
            "inputs": {"slide": {"kind": "shared_path", "path": "slide.svs"}},
            "input_access": {"kind": "shared_storage", "state": {}},
            "configuration": {"stride": 512},
            "output_path": "out.npz",
        }

        self.assertEqual(RunRequest.model_validate(body).model_dump(mode="json"), body)


if __name__ == "__main__":
    unittest.main()
