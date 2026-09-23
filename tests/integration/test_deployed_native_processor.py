from __future__ import annotations

import base64
import json
import os
import time
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx

from validation.empaia_app_service import JobView, ValidationScenario


class DeployedNativeProcessorTests(unittest.TestCase):
    def test_native_heatmap_is_published_through_empaia(self) -> None:
        job = self._published_job(ValidationScenario.NATIVE_PROCESSOR)
        sample = job.tile_sample
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertGreater(sample, 0.6)
        self.assertLess(sample, 0.9)

    def test_native_rectangle_mask_limits_the_published_heatmap(self) -> None:
        job = self._published_job(ValidationScenario.NATIVE_RECTANGLE)
        self.assertEqual(job.rectangle_reads, 1)
        sample = job.tile_sample
        background = job.tile_background_sample
        self.assertIsNotNone(sample)
        self.assertIsNotNone(background)
        assert sample is not None and background is not None
        self.assertGreater(sample, 0.6)
        self.assertLess(sample, 0.9)
        self.assertLess(background, 0.1)

    def _published_job(self, scenario: ValidationScenario) -> JobView:
        if os.environ.get("VALIDATION_NATIVE_PROCESSOR") != "1":
            self.skipTest("native processor validation profile is not active")
        scheduler_origin = os.environ["VALIDATION_ORIGIN"]
        appservice_origin = os.environ["VALIDATION_APPSERVICE_ORIGIN"]
        job_id = f"{scenario.value}-{uuid4().hex}"

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
            job = JobView.model_validate(recorded.json())
            self.assertTrue(job.finalized)
            self.assertEqual(job.tile_count, 1)
            return job
