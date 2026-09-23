from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import time
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx

from validation.empaia_app_service import ValidationScenario


if TYPE_CHECKING:
    from types import TracebackType


NAMESPACE = "peknik-ns"
RAYSERVICE = "model-service-validation"
SCHEDULER_DEPLOYMENT = "model-service-validation-execution-scheduler"


@dataclass(slots=True)
class PortForward:
    resource: str
    local_port: int
    remote_port: int
    process: subprocess.Popen[str] | None = None

    def __enter__(self) -> PortForward:
        self.process = subprocess.Popen(
            [
                "oc",
                "-n",
                NAMESPACE,
                "port-forward",
                self.resource,
                f"{self.local_port}:{self.remote_port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                if self.process.stderr is not None:
                    self.process.stderr.close()
                raise RuntimeError(f"port-forward stopped: {stderr.strip()}")
            try:
                with socket.create_connection(("127.0.0.1", self.local_port), 0.2):
                    return self
            except OSError:
                time.sleep(0.1)
        self.__exit__(None, None, None)
        raise RuntimeError(f"port-forward did not open port {self.local_port}")

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.process.stderr is not None:
            self.process.stderr.close()


class DeployedSchedulerRecoveryTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("VALIDATION_KUBERNETES_FAULTS") == "1",
        "VALIDATION_KUBERNETES_FAULTS=1 is required",
    )
    def test_replacement_scheduler_recovers_an_inflight_execution(self) -> None:
        job_id = f"{ValidationScenario.SCHEDULER_RESTART}-{uuid4().hex}"
        with (
            PortForward("service/model-service-validation-appservice", 18084, 8000),
            httpx.Client(base_url="http://127.0.0.1:18084", timeout=30) as appservice,
        ):
            with (
                PortForward(
                    "service/model-service-validation-execution-scheduler", 18083, 8080
                ),
                httpx.Client(
                    base_url="http://127.0.0.1:18083", timeout=30
                ) as scheduler,
            ):
                execution_id = _schedule(scheduler, job_id)
                _wait_for_job_field(appservice, job_id, "direct_access_reads")

            subprocess.run(
                [
                    "oc",
                    "-n",
                    NAMESPACE,
                    "delete",
                    "pod",
                    "-l",
                    (
                        "app.kubernetes.io/name=model-service-execution-scheduler,"
                        "app.kubernetes.io/instance=model-service-validation"
                    ),
                    "--wait=true",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "oc",
                    "-n",
                    NAMESPACE,
                    "rollout",
                    "status",
                    f"deployment/{SCHEDULER_DEPLOYMENT}",
                    "--timeout=180s",
                ],
                check=True,
            )

            with (
                PortForward(
                    "service/model-service-validation-execution-scheduler", 18083, 8080
                ),
                httpx.Client(
                    base_url="http://127.0.0.1:18083", timeout=30
                ) as scheduler,
            ):
                execution = _wait_for_terminal(scheduler, execution_id)

            self.assertEqual(execution["status"], "succeeded", execution)
            recorded = _wait_for_job_field(appservice, job_id, "finalized")
            self.assertEqual(recorded["configuration_reads"], 1)
            direct_access_reads = recorded["direct_access_reads"]
            self.assertIsInstance(direct_access_reads, int)
            assert isinstance(direct_access_reads, int)
            self.assertGreaterEqual(direct_access_reads, 2)
            self.assertEqual(recorded["tile_count"], 1)

    @unittest.skipUnless(
        os.environ.get("VALIDATION_KUBERNETES_FAULTS") == "1",
        "VALIDATION_KUBERNETES_FAULTS=1 is required",
    )
    def test_processing_recovers_after_ray_head_loss(self) -> None:
        job_id = f"{ValidationScenario.RAY_RESTART}-{uuid4().hex}"
        with (
            PortForward("service/model-service-validation-appservice", 18084, 8000),
            httpx.Client(base_url="http://127.0.0.1:18084", timeout=30) as appservice,
            PortForward(
                "service/model-service-validation-execution-scheduler", 18083, 8080
            ),
            httpx.Client(base_url="http://127.0.0.1:18083", timeout=30) as scheduler,
        ):
            execution_id = _schedule(scheduler, job_id)
            _wait_for_job_field(appservice, job_id, "direct_access_reads")
            cluster = subprocess.check_output(
                [
                    "oc",
                    "-n",
                    NAMESPACE,
                    "get",
                    "rayservice",
                    RAYSERVICE,
                    "-o",
                    "jsonpath={.status.activeServiceStatus.rayClusterName}",
                ],
                text=True,
            ).strip()
            self.assertTrue(cluster.startswith(f"{RAYSERVICE}-"), cluster)
            selector = f"ray.io/cluster={cluster},ray.io/node-type=head"
            subprocess.run(
                [
                    "oc",
                    "-n",
                    NAMESPACE,
                    "delete",
                    "pod",
                    "-l",
                    selector,
                    "--wait=true",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "oc",
                    "-n",
                    NAMESPACE,
                    "wait",
                    "pod",
                    "-l",
                    selector,
                    "--for=condition=Ready",
                    "--timeout=240s",
                ],
                check=True,
            )

            execution = _wait_for_terminal(scheduler, execution_id)
            self.assertEqual(execution["status"], "succeeded", execution)
            recorded = _wait_for_job_field(appservice, job_id, "finalized")
            direct_access_reads = recorded["direct_access_reads"]
            self.assertIsInstance(direct_access_reads, int)
            assert isinstance(direct_access_reads, int)
            self.assertGreaterEqual(direct_access_reads, 2)
            self.assertEqual(recorded["tile_count"], 1)


def _schedule(client: httpx.Client, job_id: str) -> str:
    admitted = client.post(
        "/v1/providers/empaia/workflows/prostate-heatmap/schedule",
        json={
            "installation_id": "validation",
            "api_url": (
                "http://model-service-validation-appservice."
                "peknik-ns.svc.cluster.local:8000"
            ),
            "job_id": job_id,
            "credential": _credential(),
        },
    )
    if admitted.status_code != 202:
        raise AssertionError(admitted.text)
    return admitted.json()["execution"]["id"]


def _wait_for_job_field(
    client: httpx.Client, job_id: str, field: str
) -> dict[str, object]:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        response = client.get(f"/validation/jobs/{job_id}")
        if response.status_code == 200 and response.json().get(field):
            return response.json()
        if response.status_code not in {200, 404}:
            raise AssertionError(response.text)
        time.sleep(0.5)
    raise AssertionError(f"AppService did not record {field}")


def _wait_for_terminal(client: httpx.Client, execution_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        response = client.get(f"/v1/executions/{execution_id}")
        if response.status_code != 200:
            raise AssertionError(response.text)
        execution = response.json()
        if execution["status"] in {"succeeded", "failed", "cancelled"}:
            return execution
        time.sleep(1)
    raise AssertionError("execution did not reach a terminal status")


def _credential() -> str:
    def encode(value: dict[str, object]) -> str:
        return (
            base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
        )

    expiry = int((datetime.now(UTC) + timedelta(minutes=10)).timestamp())
    return f"{encode({})}.{encode({'exp': expiry})}.synthetic-signature"


if __name__ == "__main__":
    unittest.main()
