from __future__ import annotations

import base64
import json
import os
import time
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx

from validation.empaia_app_service import ValidationScenario


class DeployedNativeProcessorTests(unittest.TestCase):
    def test_native_heatmap_is_published_through_empaia(self) -> None:
        if os.environ.get("VALIDATION_NATIVE_PROCESSOR") != "1":
            self.skipTest("native processor validation profile is not active")
        scheduler_origin = os.environ["VALIDATION_ORIGIN"]
        appservice_origin = os.environ["VALIDATION_APPSERVICE_ORIGIN"]
        job_id = f"{ValidationScenario.NATIVE_PROCESSOR.value}-{uuid4().hex}"

        def encode(value: dict[str, object]) -> str:
            return (
                base64.urlsafe_b64encode(json.dumps(value).encode())
                .rstrip(b"=")
                .decode()
            )

        expiry = int((datetime.now(UTC) + timedelta(minutes=10)).timestamp())
        credential = f"{encode({})}.{encode({'exp': expiry})}.synthetic-signature"
        with (
            httpx.Client(base_url=scheduler_origin, timeout=30) as scheduler,
            httpx.Client(base_url=appservice_origin, timeout=30) as appservice,
        ):
            admitted = scheduler.post(
                "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
                json={
                    "installation_id": "validation",
                    "api_url": os.environ.get(
                        "VALIDATION_APPSERVICE_CLUSTER_URL",
                        "http://model-service-validation-appservice.peknik-ns.svc.cluster.local:8000",
                    ),
                    "job_id": job_id,
                    "credential": credential,
                },
            )
            self.assertEqual(admitted.status_code, 202, admitted.text)
            execution_id = admitted.json()["execution"]["id"]

            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                response = scheduler.get(f"/v1/executions/{execution_id}")
                self.assertEqual(response.status_code, 200, response.text)
                execution = response.json()
                if execution["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(1)
            else:
                self.fail("native processor execution did not finish")
            self.assertEqual(execution["status"], "succeeded", execution)

            recorded = appservice.get(f"/validation/jobs/{job_id}")
            self.assertEqual(recorded.status_code, 200, recorded.text)
            job = recorded.json()
            self.assertTrue(job["finalized"])
            self.assertEqual(job["tile_count"], 1)
            self.assertGreater(job["tile_sample"], 0.6)
            self.assertLess(job["tile_sample"], 0.9)
