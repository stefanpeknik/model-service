from __future__ import annotations

import os
import time
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx


class DeployedSchedulerTests(unittest.TestCase):
    def test_deadline_fails_execution_without_ray_capacity(self) -> None:
        origin = os.environ.get("VALIDATION_ORIGIN")
        if origin is None or os.environ.get("VALIDATION_NO_WORKER") != "1":
            self.skipTest("validation scheduler must run with no Ray workers")

        with httpx.Client(base_url=origin, timeout=30) as client:
            admitted = client.post(
                "/v1/workflows/prostate-heatmap/schedule",
                json={
                    "provider": "standalone",
                    "body": self._invocation(f"output/{uuid4().hex}.tiff"),
                    "idempotency_key": uuid4().hex,
                    "requested_timeout_seconds": 5,
                },
            )
            self.assertEqual(admitted.status_code, 202, admitted.text)
            execution_id = admitted.json()["execution"]["id"]

            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                status = client.get(f"/v1/executions/{execution_id}")
                self.assertEqual(status.status_code, 200, status.text)
                if status.json()["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(1)
            else:
                self.fail("expired execution did not reach a terminal status")

            self.assertEqual(status.json()["status"], "failed", status.text)
            self.assertEqual(
                status.json()["error"]["code"], "execution_deadline_exceeded"
            )
            self.assertIsNone(status.json()["result"])

    def test_cancellation_remains_terminal_without_ray_capacity(self) -> None:
        origin = os.environ.get("VALIDATION_ORIGIN")
        if origin is None or os.environ.get("VALIDATION_NO_WORKER") != "1":
            self.skipTest("validation scheduler must run with no Ray workers")

        with httpx.Client(base_url=origin, timeout=30) as client:
            admitted = client.post(
                "/v1/workflows/prostate-heatmap/schedule",
                json={
                    "provider": "standalone",
                    "body": self._invocation(f"output/{uuid4().hex}.tiff"),
                    "idempotency_key": uuid4().hex,
                },
            )
            self.assertEqual(admitted.status_code, 202, admitted.text)
            execution_id = admitted.json()["execution"]["id"]

            time.sleep(15)
            waiting = client.get(f"/v1/executions/{execution_id}")
            self.assertEqual(waiting.status_code, 200, waiting.text)
            self.assertIn(waiting.json()["status"], {"queued", "running"})
            self.assertIsNone(waiting.json()["result"])

            cancelled = client.post(f"/v1/executions/{execution_id}/cancel")
            self.assertEqual(cancelled.status_code, 200, cancelled.text)
            self.assertEqual(cancelled.json()["status"], "cancelled")
            repeated = client.post(f"/v1/executions/{execution_id}/cancel")
            self.assertEqual(repeated.status_code, 200, repeated.text)
            self.assertEqual(repeated.json()["status"], "cancelled")

    def test_invalid_public_requests_return_safe_errors(self) -> None:
        origin = os.environ.get("VALIDATION_ORIGIN")
        if origin is None:
            self.skipTest("VALIDATION_ORIGIN is not configured")

        request = {
            "provider": "standalone",
            "body": self._invocation(f"output/{uuid4().hex}.tiff"),
            "idempotency_key": uuid4().hex,
        }
        with httpx.Client(base_url=origin, timeout=30) as client:
            unknown_workflow = client.post(
                "/v1/workflows/not-deployed/schedule", json=request
            )
            self.assertEqual(unknown_workflow.status_code, 404)
            self.assertEqual(
                unknown_workflow.json()["detail"]["code"], "unknown_workflow"
            )

            invalid_body = client.post(
                "/v1/workflows/prostate-heatmap/schedule",
                json={
                    **request,
                    "body": {
                        **request["body"],
                        "input_access": {
                            "kind": "shared_storage",
                            "state": {"root": "/missing"},
                        },
                    },
                },
            )
            self.assertEqual(invalid_body.status_code, 422)
            self.assertEqual(
                invalid_body.json()["detail"]["code"],
                "invalid_schedule_request",
            )

            unknown_execution = client.get(f"/v1/executions/{uuid4()}")
            self.assertEqual(unknown_execution.status_code, 404)
            self.assertEqual(
                unknown_execution.json()["detail"]["code"], "unknown_execution"
            )

    def test_replay_returns_the_same_execution_and_conflict_is_rejected(self) -> None:
        origin = os.environ.get("VALIDATION_ORIGIN")
        if origin is None:
            self.skipTest("VALIDATION_ORIGIN is not configured")

        key = uuid4().hex
        request = {
            "provider": "standalone",
            "body": self._invocation(f"output/{uuid4().hex}.tiff"),
            "idempotency_key": key,
        }
        with httpx.Client(base_url=origin, timeout=30) as client:
            admitted = client.post(
                "/v1/workflows/prostate-heatmap/schedule", json=request
            )
            self.assertEqual(admitted.status_code, 202, admitted.text)

            replay = client.post(
                "/v1/workflows/prostate-heatmap/schedule", json=request
            )
            self.assertEqual(replay.status_code, 200, replay.text)
            self.assertEqual(
                replay.json()["execution"]["id"],
                admitted.json()["execution"]["id"],
            )

            conflict = client.post(
                "/v1/workflows/prostate-heatmap/schedule",
                json={
                    **request,
                    "body": self._invocation(f"output/{uuid4().hex}.tiff"),
                },
            )
            self.assertEqual(conflict.status_code, 409, conflict.text)

    def test_standalone_schedule_completes_with_the_synchronous_result(self) -> None:
        origin = os.environ.get("VALIDATION_ORIGIN")
        if origin is None:
            self.skipTest("VALIDATION_ORIGIN is not configured")

        invocation = self._invocation(f"output/{uuid4().hex}.tiff")
        with httpx.Client(base_url=origin, timeout=30) as client:
            admitted = client.post(
                "/v1/workflows/prostate-heatmap/schedule",
                json={
                    "provider": "standalone",
                    "body": invocation,
                    "idempotency_key": uuid4().hex,
                },
            )
            self.assertEqual(admitted.status_code, 202, admitted.text)
            execution_id = admitted.json()["execution"]["id"]

            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                status = client.get(f"/v1/executions/{execution_id}")
                self.assertEqual(status.status_code, 200, status.text)
                if status.json()["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(1)
            else:
                self.fail("scheduled execution did not reach a terminal status")

            self.assertEqual(status.json()["status"], "succeeded", status.text)
            synchronous = client.post(
                "/v1/workflows/prostate-heatmap/run",
                json={
                    **invocation,
                    "deadline_at": (
                        datetime.now(UTC) + timedelta(minutes=5)
                    ).isoformat(),
                },
            )
            self.assertEqual(synchronous.status_code, 200, synchronous.text)
            self.assertEqual(status.json()["result"], synchronous.json())

    @staticmethod
    def _invocation(output_path: str) -> dict[str, object]:
        return {
            "inputs": {
                "slide": {"kind": "shared_path", "path": "input/slide.dat"},
                "region_of_interest": {
                    "kind": "shared_path",
                    "path": "input/roi.dat",
                },
            },
            "resolved_configuration": {
                "output_path": output_path,
                "stride_fraction": 0.5,
                "output_tile_height": 256,
                "output_tile_width": 256,
            },
            "input_access": {
                "kind": "shared_storage",
                "state": {"root": "/mnt/validation"},
            },
        }


if __name__ == "__main__":
    unittest.main()
