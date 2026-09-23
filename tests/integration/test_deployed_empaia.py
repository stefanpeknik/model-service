from __future__ import annotations

import base64
import json
import os
import time
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx


class DeployedEmpaiaTests(unittest.TestCase):
    def test_public_empaia_schedule_publishes_and_finalizes(self) -> None:
        origin = os.environ.get("VALIDATION_ORIGIN")
        appservice_origin = os.environ.get("VALIDATION_APPSERVICE_ORIGIN")
        if origin is None or appservice_origin is None:
            self.skipTest("validation scheduler and AppService origins are required")

        job_id = uuid4().hex
        credential = self._credential()
        request = {
            "installation_id": "validation",
            "api_url": os.environ.get(
                "VALIDATION_APPSERVICE_CLUSTER_URL",
                "http://model-service-validation-appservice.peknik-ns.svc.cluster.local:8000",
            ),
            "job_id": job_id,
            "credential": credential,
        }
        with (
            httpx.Client(base_url=origin, timeout=30) as scheduler,
            httpx.Client(base_url=appservice_origin, timeout=30) as appservice,
        ):
            admitted = scheduler.post(
                "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
                json=request,
            )
            self.assertEqual(admitted.status_code, 202, admitted.text)
            execution_id = admitted.json()["execution"]["id"]
            replay = scheduler.post(
                "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
                json=request,
            )
            self.assertEqual(replay.status_code, 200, replay.text)
            self.assertEqual(replay.json()["execution"]["id"], execution_id)

            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                response = scheduler.get(f"/v1/executions/{execution_id}")
                self.assertEqual(response.status_code, 200, response.text)
                if response.json()["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(1)
            else:
                self.fail("EMPAIA execution did not reach a terminal status")

            self.assertEqual(response.json()["status"], "succeeded", response.text)
            recorded = appservice.get(f"/validation/jobs/{job_id}")
            self.assertEqual(recorded.status_code, 200, recorded.text)
            self.assertEqual(recorded.json()["configuration_reads"], 1)
            self.assertEqual(recorded.json()["tile_count"], 1)
            self.assertTrue(recorded.json()["finalized"])
            self.assertIsNone(recorded.json()["failure"])

    def test_rectangle_input_is_materialized_before_publication(self) -> None:
        origin = os.environ.get("VALIDATION_ORIGIN")
        appservice_origin = os.environ.get("VALIDATION_APPSERVICE_ORIGIN")
        if origin is None or appservice_origin is None:
            self.skipTest("validation scheduler and AppService origins are required")

        job_id = f"rectangle-{uuid4().hex}"
        request = {
            "installation_id": "validation",
            "api_url": os.environ.get(
                "VALIDATION_APPSERVICE_CLUSTER_URL",
                "http://model-service-validation-appservice.peknik-ns.svc.cluster.local:8000",
            ),
            "job_id": job_id,
            "credential": self._credential(),
        }
        with (
            httpx.Client(base_url=origin, timeout=30) as scheduler,
            httpx.Client(base_url=appservice_origin, timeout=30) as appservice,
        ):
            admitted = scheduler.post(
                "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
                json=request,
            )
            self.assertEqual(admitted.status_code, 202, admitted.text)
            execution_id = admitted.json()["execution"]["id"]
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                response = scheduler.get(f"/v1/executions/{execution_id}")
                self.assertEqual(response.status_code, 200, response.text)
                if response.json()["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(1)
            else:
                self.fail("rectangle execution did not reach a terminal status")

            self.assertEqual(response.json()["status"], "succeeded", response.text)
            recorded = appservice.get(f"/validation/jobs/{job_id}")
            self.assertEqual(recorded.status_code, 200, recorded.text)
            self.assertEqual(recorded.json()["configuration_reads"], 1)
            self.assertEqual(recorded.json()["rectangle_reads"], 1)
            self.assertEqual(recorded.json()["tile_count"], 1)
            self.assertTrue(recorded.json()["finalized"])

    @staticmethod
    def _credential() -> str:
        def encode(value: dict[str, object]) -> str:
            return (
                base64.urlsafe_b64encode(json.dumps(value).encode())
                .rstrip(b"=")
                .decode()
            )

        expiry = int((datetime.now(UTC) + timedelta(minutes=10)).timestamp())
        return f"{encode({})}.{encode({'exp': expiry})}.synthetic-signature"


if __name__ == "__main__":
    unittest.main()
