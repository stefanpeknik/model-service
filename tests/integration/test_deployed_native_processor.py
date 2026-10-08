"""Check live native slide processing and pixelmap publication against synthetic image fixtures.

These tests contact configured live services and may create executions and artifacts.
They are not part of offline unit validation.
"""

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
    """Exercise live native slide processing and pixelmap publication against synthetic image fixtures."""

    def test_native_heatmap_is_published_through_empaia(self) -> None:
        """Verify native heatmap is published through EMPAIA.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        job = self._published_job(ValidationScenario.NATIVE_PROCESSOR)
        sample = job.tile_sample
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertGreater(sample, 0.6)
        self.assertLess(sample, 0.9)

    def test_native_rectangle_mask_limits_the_published_heatmap(self) -> None:
        """Verify native rectangle mask limits the published heatmap.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Submit a synthetic-image EMPAIA job to the native validation profile and return its observed published job state.

        Args:
            scenario: Validation job prefix selecting deterministic mock behavior.

        Returns:
            JobView: Observable mock call counters, tile samples, and terminal job
                state.
        """
        if os.environ.get("VALIDATION_NATIVE_PROCESSOR") != "1":
            self.skipTest("native processor validation profile is not active")
        scheduler_origin = os.environ["VALIDATION_ORIGIN"]
        appservice_origin = os.environ["VALIDATION_APPSERVICE_ORIGIN"]
        job_id = f"{scenario.value}-{uuid4().hex}"

        def encode(value: dict[str, object]) -> str:
            """Encode a synthetic credential section as unpadded URL-safe base64 JSON.

            Args:
                value: Value being validated, serialized, converted, or checked by this
                    helper.

            Returns:
                str: Unpadded URL-safe base64 JSON section for a synthetic credential.
            """
            return (
                base64.urlsafe_b64encode(json.dumps(value).encode())
                .rstrip(b"=")
                .decode()
            )

        expiry = int((datetime.now(UTC) + timedelta(minutes=10)).timestamp())
        signature = (
            base64.urlsafe_b64encode(b"synthetic-signature").rstrip(b"=").decode()
        )
        credential = f"{encode({})}.{encode({'exp': expiry})}.{signature}"
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
