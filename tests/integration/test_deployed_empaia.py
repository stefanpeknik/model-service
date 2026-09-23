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


class DeployedEmpaiaTests(unittest.TestCase):
    def test_public_empaia_schedule_publishes_and_finalizes(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.DIRECT_ACCESS)
            execution_id = self._schedule(scheduler, request, replay=True)
            execution = self._wait_terminal(scheduler, execution_id)
            self.assertEqual(execution["status"], "succeeded", execution)

            recorded = self._wait_job(appservice, job_id, "finalized")
            self.assertEqual(recorded["configuration_reads"], 1)
            self.assertEqual(recorded["tile_count"], 1)
            self.assertTrue(recorded["finalized"])
            self.assertIsNone(recorded["failure"])

    def test_job_replay_accepts_new_credential_but_rejects_changed_origin(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.DIRECT_ACCESS)
            execution_id = self._schedule(scheduler, request)
            path = "/v1/providers/empaia/workflows/prostate-heatmap/schedule"

            replacement = scheduler.post(
                path,
                json={**request, "credential": self._credential(valid_seconds=700)},
            )
            self.assertEqual(replacement.status_code, 200, replacement.text)
            self.assertEqual(replacement.json()["execution"]["id"], execution_id)

            changed_origin = scheduler.post(
                path,
                json={**request, "api_url": f"{request['api_url']}/alternate"},
            )
            self.assertEqual(changed_origin.status_code, 409, changed_origin.text)
            self.assertEqual(
                changed_origin.json()["detail"]["code"], "schedule_conflict"
            )

            execution = self._wait_terminal(scheduler, execution_id)
            self.assertEqual(execution["status"], "succeeded", execution)
            recorded = self._wait_job(appservice, job_id, "finalized")
            self.assertEqual(recorded["configuration_reads"], 1)
            self.assertEqual(recorded["tile_count"], 1)
            self.assertEqual(recorded["finalize_attempts"], 1)

    def test_rectangle_input_is_materialized_before_publication(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.RECTANGLE)
            execution_id = self._schedule(scheduler, request)
            execution = self._wait_terminal(scheduler, execution_id)
            self.assertEqual(execution["status"], "succeeded", execution)

            recorded = self._wait_job(appservice, job_id, "finalized")
            self.assertEqual(recorded["configuration_reads"], 1)
            self.assertEqual(recorded["rectangle_reads"], 1)
            self.assertEqual(recorded["tile_count"], 1)

    def test_invalid_configuration_is_rejected_before_admission(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.INVALID_CONFIGURATION)
            response = scheduler.post(
                "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
                json=request,
            )
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(
                response.json()["detail"]["code"], "invalid_schedule_request"
            )
            recorded = self._wait_job(appservice, job_id, "configuration_reads")
            self.assertEqual(recorded["configuration_reads"], 1)
            self.assertEqual(recorded["direct_access_reads"], 0)

    def test_rejected_credential_is_a_permanent_admission_error(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.AUTH_REJECTED)
            response = scheduler.post(
                "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
                json=request,
            )
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(
                response.json()["detail"]["code"], "invalid_schedule_request"
            )
            recorded = self._wait_job(appservice, job_id, "configuration_reads")
            self.assertEqual(recorded["configuration_reads"], 1)

    def test_per_workflow_limit_queues_then_releases_capacity(self) -> None:
        with self._clients() as (scheduler, appservice):
            first_job_id, first_request = self._request(ValidationScenario.CANCELLATION)
            first_id = self._schedule(scheduler, first_request)
            self._wait_status(scheduler, first_id, "running")

            _, second_request = self._request(ValidationScenario.DIRECT_ACCESS)
            second_id = self._schedule(scheduler, second_request)
            queued = scheduler.get(f"/v1/executions/{second_id}")
            self.assertEqual(queued.status_code, 200, queued.text)
            self.assertEqual(queued.json()["status"], "queued")

            cancelled = scheduler.post(f"/v1/executions/{first_id}/cancel")
            self.assertEqual(cancelled.status_code, 200, cancelled.text)
            self.assertEqual(cancelled.json()["status"], "cancelled")
            self._wait_job(appservice, first_job_id, "failure_reports")

            second = self._wait_terminal(scheduler, second_id)
            self.assertEqual(second["status"], "succeeded", second)

    def test_temporary_appservice_failure_rejects_admission_as_retryable(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.SERVICE_UNAVAILABLE)
            response = scheduler.post(
                "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
                json=request,
            )
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(
                response.json()["detail"]["code"], "scheduling_unavailable"
            )
            recorded = self._wait_job(appservice, job_id, "configuration_reads")
            self.assertEqual(recorded["configuration_reads"], 1)

    def test_unknown_tile_upload_outcome_is_not_replayed(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.PUBLICATION_FAILURE)
            execution_id = self._schedule(scheduler, request)
            execution = self._wait_terminal(scheduler, execution_id)
            self.assertEqual(execution["status"], "failed", execution)
            error = execution["error"]
            self.assertIsInstance(error, dict)
            assert isinstance(error, dict)
            self.assertEqual(error["code"], "empaia_finish_failed")

            recorded = self._wait_job(appservice, job_id, "tile_attempts")
            self.assertEqual(recorded["tile_attempts"], 1)
            self.assertEqual(recorded["tile_count"], 0)
            self.assertEqual(recorded["finalize_attempts"], 0)
            self.assertFalse(recorded["finalized"])

    def test_unknown_finalization_outcome_is_not_replayed(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.FINALIZATION_FAILURE)
            execution_id = self._schedule(scheduler, request)
            execution = self._wait_terminal(scheduler, execution_id)
            self.assertEqual(execution["status"], "failed", execution)
            error = execution["error"]
            self.assertIsInstance(error, dict)
            assert isinstance(error, dict)
            self.assertEqual(error["code"], "empaia_finish_failed")

            recorded = self._wait_job(appservice, job_id, "finalize_attempts")
            self.assertEqual(recorded["tile_attempts"], 1)
            self.assertEqual(recorded["tile_count"], 1)
            self.assertEqual(recorded["finalize_attempts"], 1)
            self.assertFalse(recorded["finalized"])

    def test_workflow_failure_is_reported_to_appservice(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.WORKFLOW_FAILURE)
            execution_id = self._schedule(scheduler, request)
            execution = self._wait_terminal(scheduler, execution_id)
            self.assertEqual(execution["status"], "failed", execution)
            error = execution["error"]
            self.assertIsInstance(error, dict)
            assert isinstance(error, dict)
            self.assertEqual(error["code"], "workflow_internal_error")

            recorded = self._wait_job(appservice, job_id, "failure_reports")
            self.assertEqual(recorded["failure_reports"], 1)
            direct_access_reads = recorded["direct_access_reads"]
            self.assertIsInstance(direct_access_reads, int)
            assert isinstance(direct_access_reads, int)
            self.assertGreaterEqual(direct_access_reads, 3)
            self.assertEqual(recorded["failure"], "workflow failed unexpectedly")
            self.assertFalse(recorded["finalized"])

    def test_cancellation_is_reported_to_appservice(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.CANCELLATION)
            execution_id = self._schedule(scheduler, request)
            running = self._wait_status(scheduler, execution_id, "running")
            self.assertEqual(running["status"], "running")

            cancelled = scheduler.post(f"/v1/executions/{execution_id}/cancel")
            self.assertEqual(cancelled.status_code, 200, cancelled.text)
            self.assertEqual(cancelled.json()["status"], "cancelled")
            repeated = scheduler.post(f"/v1/executions/{execution_id}/cancel")
            self.assertEqual(repeated.status_code, 200, repeated.text)
            self.assertEqual(repeated.json()["status"], "cancelled")

            recorded = self._wait_job(appservice, job_id, "failure_reports")
            self.assertEqual(recorded["failure_reports"], 1)
            self.assertEqual(recorded["failure"], "workflow execution was cancelled")
            self.assertFalse(recorded["finalized"])

    def test_credential_expiry_caps_a_blocked_execution(self) -> None:
        with self._clients() as (scheduler, appservice):
            job_id, request = self._request(ValidationScenario.CANCELLATION)
            request["credential"] = self._credential(valid_seconds=20)
            execution_id = self._schedule(scheduler, request)
            execution = self._wait_terminal(scheduler, execution_id)
            self.assertEqual(execution["status"], "failed", execution)
            error = execution["error"]
            self.assertIsInstance(error, dict)
            assert isinstance(error, dict)
            self.assertEqual(error["code"], "execution_deadline_exceeded")

            recorded = self._wait_job(appservice, job_id, "configuration_reads")
            self.assertFalse(recorded["finalized"])
            self.assertEqual(recorded["tile_count"], 0)

    def _clients(self) -> _ClientPair:
        origin = os.environ.get("VALIDATION_ORIGIN")
        appservice_origin = os.environ.get("VALIDATION_APPSERVICE_ORIGIN")
        if origin is None or appservice_origin is None:
            self.skipTest("validation scheduler and AppService origins are required")
        return _ClientPair(origin, appservice_origin)

    def _request(self, scenario: ValidationScenario) -> tuple[str, dict[str, object]]:
        job_id = f"{scenario.value}-{uuid4().hex}"
        return job_id, {
            "installation_id": "validation",
            "api_url": os.environ.get(
                "VALIDATION_APPSERVICE_CLUSTER_URL",
                "http://model-service-validation-appservice.peknik-ns.svc.cluster.local:8000",
            ),
            "job_id": job_id,
            "credential": self._credential(),
        }

    def _schedule(
        self,
        scheduler: httpx.Client,
        request: dict[str, object],
        *,
        replay: bool = False,
    ) -> str:
        path = "/v1/providers/empaia/workflows/prostate-heatmap/schedule"
        admitted = scheduler.post(path, json=request)
        self.assertEqual(admitted.status_code, 202, admitted.text)
        execution_id = admitted.json()["execution"]["id"]
        if replay:
            repeated = scheduler.post(path, json=request)
            self.assertEqual(repeated.status_code, 200, repeated.text)
            self.assertEqual(repeated.json()["execution"]["id"], execution_id)
        return execution_id

    def _wait_terminal(
        self, scheduler: httpx.Client, execution_id: str
    ) -> dict[str, object]:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            response = scheduler.get(f"/v1/executions/{execution_id}")
            self.assertEqual(response.status_code, 200, response.text)
            execution = response.json()
            if execution["status"] in {"succeeded", "failed", "cancelled"}:
                return execution
            time.sleep(1)
        self.fail("EMPAIA execution did not reach a terminal status")

    def _wait_status(
        self, scheduler: httpx.Client, execution_id: str, expected: str
    ) -> dict[str, object]:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            response = scheduler.get(f"/v1/executions/{execution_id}")
            self.assertEqual(response.status_code, 200, response.text)
            execution = response.json()
            if execution["status"] == expected:
                return execution
            if execution["status"] in {"succeeded", "failed", "cancelled"}:
                self.fail(f"execution reached {execution['status']} before {expected}")
            time.sleep(1)
        self.fail(f"EMPAIA execution did not reach {expected}")

    def _wait_job(
        self, appservice: httpx.Client, job_id: str, field: str
    ) -> dict[str, object]:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            response = appservice.get(f"/validation/jobs/{job_id}")
            if response.status_code == 200 and response.json().get(field):
                return response.json()
            self.assertIn(response.status_code, {200, 404}, response.text)
            time.sleep(0.5)
        self.fail(f"AppService did not record {field}")

    @staticmethod
    def _credential(valid_seconds: int = 600) -> str:
        def encode(value: dict[str, object]) -> str:
            return (
                base64.urlsafe_b64encode(json.dumps(value).encode())
                .rstrip(b"=")
                .decode()
            )

        expiry = int((datetime.now(UTC) + timedelta(seconds=valid_seconds)).timestamp())
        return f"{encode({})}.{encode({'exp': expiry})}.synthetic-signature"


class _ClientPair:
    def __init__(self, scheduler_origin: str, appservice_origin: str) -> None:
        self._scheduler = httpx.Client(base_url=scheduler_origin, timeout=30)
        self._appservice = httpx.Client(base_url=appservice_origin, timeout=30)

    def __enter__(self) -> tuple[httpx.Client, httpx.Client]:
        return self._scheduler, self._appservice

    def __exit__(self, *_args: object) -> None:
        self._scheduler.close()
        self._appservice.close()


if __name__ == "__main__":
    unittest.main()
