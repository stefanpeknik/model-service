"""Check live native standalone scheduling, replay/conflict handling, immediate execution, and output contracts.

These tests contact configured live services and may create executions and artifacts.
They are not part of offline unit validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4

import httpx


if TYPE_CHECKING:
    from rationai.types.workflows import WorkflowRequest


_WORKFLOW = "prostate-heatmap"
_TERMINAL = {"succeeded", "failed", "cancelled"}
_SLIDE = (
    "wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/67df0411-395d-5d73-9c2f-3e0ffef45491.mrxs"
)
_MASK = "wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/tissue-mask-open-slide.tiff"


@dataclass(frozen=True, slots=True)
class NativeWorkflowSettings:
    """Validate live test origins, shared input paths, and optional local artifact access."""

    origin: str
    storage_root: str
    slide_path: str
    mask_path: str
    completion_timeout_seconds: int
    artifact_root: Path | None

    @classmethod
    def from_environment(cls) -> NativeWorkflowSettings:
        """Validate a public origin, shared paths, and optional artifact root from live test settings.

        Returns:
            NativeWorkflowSettings: Validated live HTTP/storage inputs and optional
                artifact access settings.

        Raises:
            ValueError: A value fails the validation described above.
        """
        origin = os.environ.get("VALIDATION_ORIGIN", "").strip()
        url = urlsplit(origin)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError("VALIDATION_ORIGIN must be a public HTTP origin")
        storage_root = os.environ.get("VALIDATION_STORAGE_ROOT", "/mnt/validation")
        if not PurePosixPath(storage_root).is_absolute():
            raise ValueError("VALIDATION_STORAGE_ROOT must be absolute")
        slide_path = os.environ.get("VALIDATION_SLIDE_PATH", _SLIDE)
        mask_path = os.environ.get("VALIDATION_MASK_PATH", _MASK)
        for name, value in (("slide", slide_path), ("mask", mask_path)):
            if (
                not value
                or PurePosixPath(value).is_absolute()
                or any(part in {"", ".", ".."} for part in value.split("/"))
                or "\\" in value
            ):
                raise ValueError(f"{name} must be a traversal-free relative path")
        timeout = int(os.environ.get("VALIDATION_COMPLETION_TIMEOUT_SECONDS", "900"))
        if not 30 <= timeout <= 1800:
            raise ValueError("completion timeout must be between 30 and 1800 seconds")
        artifact_root = os.environ.get("VALIDATION_ARTIFACT_ROOT")
        return cls(
            origin=origin.rstrip("/"),
            storage_root=storage_root,
            slide_path=slide_path,
            mask_path=mask_path,
            completion_timeout_seconds=timeout,
            artifact_root=Path(artifact_root).resolve(strict=True)
            if artifact_root
            else None,
        )

    def invocation(self, output_path: str) -> dict[str, object]:
        """Build a JSON shared-storage invocation using the configured real slide/mask and requested output path.

        Args:
            output_path: Destination artifact path or configured relative output path.

        Returns:
            dict[str, object]: Decoded, serialized, or recorded fields described above.
        """
        return {
            "inputs": {
                "slide": {"kind": "shared_path", "path": self.slide_path},
                "region_of_interest": {"kind": "shared_path", "path": self.mask_path},
            },
            "resolved_configuration": {
                "output_path": output_path,
                "stride_fraction": 0.5,
                "output_tile_height": 256,
                "output_tile_width": 256,
            },
            "input_access": {
                "kind": "shared_storage",
                "state": {"root": self.storage_root},
            },
        }

    def sdk_request(self, output_path: str) -> WorkflowRequest:
        """Build the equivalent typed SDK request using the same inputs and resolved heatmap configuration.

        Args:
            output_path: Destination artifact path or configured relative output path.

        Returns:
            WorkflowRequest: Typed SDK request using the selected shared-storage fixture
                inputs.
        """
        from rationai.types.workflows import (
            InputAccess,
            SharedPathInput,
            WorkflowRequest,
        )

        return WorkflowRequest(
            inputs={
                "slide": SharedPathInput(self.slide_path),
                "region_of_interest": SharedPathInput(self.mask_path),
            },
            resolved_configuration={
                "output_path": output_path,
                "stride_fraction": 0.5,
                "output_tile_height": 256,
                "output_tile_width": 256,
            },
            input_access=InputAccess(
                kind="shared_storage", state={"root": self.storage_root}
            ),
        )


def _evidence(case: str, **values: object) -> None:
    """Print a structured live-test evidence record with a stable prefix for later validation reports.

    Args:
        case: Stable live-validation case name included in evidence output.
        **values: Keyword arguments forwarded to the delegated operation or accepted by
            the fixture interface.

    Returns:
        None: No return value.
    """
    print(
        "NATIVE_WORKFLOW_EVIDENCE "
        + json.dumps({"case": case, **values}, sort_keys=True)
    )


@unittest.skipUnless(
    os.environ.get("RUN_DEPLOYED_NATIVE_WORKFLOWS") == "1",
    "RUN_DEPLOYED_NATIVE_WORKFLOWS=1 is required for real WSI validation",
)
class DeployedNativeWorkflowTests(unittest.TestCase):
    """Exercise live native standalone scheduling, replay/conflict handling, immediate execution, and output contracts."""

    def setUp(self) -> None:
        """Create isolated fixtures and clients for live native standalone scheduling, replay/conflict handling, immediate execution, and output contracts.

        Returns:
            None: No return value.
        """
        self.settings = NativeWorkflowSettings.from_environment()
        self.http = httpx.Client(base_url=self.settings.origin, timeout=30)
        self.addCleanup(self.http.close)

    def test_unknown_workflow_and_execution_return_safe_errors(self) -> None:
        """Verify unknown workflow and execution return safe errors.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        response = self.http.post(
            "/v1/workflows/validation-unknown-workflow/schedule",
            json={
                "provider": "standalone",
                "body": self.settings.invocation(f"outputs/{uuid4().hex}.tiff"),
                "idempotency_key": uuid4().hex,
            },
        )
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(response.json()["detail"]["code"], "unknown_workflow")
        self.assertFalse(response.json()["detail"]["retryable"])
        missing = self.http.get(f"/v1/executions/{uuid4()}")
        self.assertEqual(missing.status_code, 404, missing.text)
        self.assertEqual(missing.json()["detail"]["code"], "unknown_execution")
        _evidence(
            "unknown_identifiers", workflow=response.json(), execution=missing.json()
        )

    @unittest.skipIf(
        os.environ.get("VALIDATION_SCHEDULER_WORK_PAUSED") == "1",
        "successful processing requires scheduler claims enabled",
    )
    def test_schedule_replay_conflict_and_immediate_run_share_output_contract(
        self,
    ) -> None:
        """Verify schedule replay conflict and immediate run share output contract.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        from rationai import Client
        from rationai.errors import SchedulingUnavailableError
        from rationai.types.workflows import ExecutionStatus, SharedArtifact

        key = f"native-full-wsi-{uuid4().hex}"
        output_path = f"outputs/{key}.tiff"
        request = self.settings.sdk_request(output_path)
        schedule_body = {
            "provider": "standalone",
            "body": request.to_dict(),
            "idempotency_key": key,
            "requested_timeout_seconds": self.settings.completion_timeout_seconds,
        }
        _evidence("success_request", request=schedule_body)
        admitted = self.http.post(
            f"/v1/workflows/{_WORKFLOW}/schedule", json=schedule_body
        )
        self.assertEqual(admitted.status_code, 202, admitted.text)
        admission = admitted.json()
        self.assertTrue(admission["created"])
        execution_id = admission["execution"]["id"]
        self.addCleanup(self._cancel_unfinished, execution_id)
        self._assert_public_execution(admission["execution"], execution_id)
        _evidence("admission", response=admission)

        replay = self.http.post(
            f"/v1/workflows/{_WORKFLOW}/schedule", json=schedule_body
        )
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertFalse(replay.json()["created"])
        self.assertEqual(replay.json()["execution"]["id"], execution_id)
        conflict = self.http.post(
            f"/v1/workflows/{_WORKFLOW}/schedule",
            json={
                **schedule_body,
                "requested_timeout_seconds": (
                    self.settings.completion_timeout_seconds - 1
                ),
            },
        )
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertEqual(conflict.json()["detail"]["code"], "schedule_conflict")
        self.assertFalse(conflict.json()["detail"]["retryable"])
        _evidence("replay_and_conflict", replay=replay.json(), conflict=conflict.json())

        with Client(base_url=self.settings.origin, timeout=30) as client:
            deadline = time.monotonic() + self.settings.completion_timeout_seconds + 30
            while time.monotonic() < deadline:
                try:
                    current = client.executions.get(execution_id)
                except (SchedulingUnavailableError, httpx.TransportError) as error:
                    _evidence("poll_unavailable", error=type(error).__name__)
                    time.sleep(1)
                    continue
                public = asdict(current)
                self._assert_public_execution(public, execution_id)
                _evidence("poll", execution=public)
                if current.status.value in _TERMINAL:
                    break
                time.sleep(1)
            else:
                self.fail(
                    f"execution {execution_id} did not finish before polling limit"
                )
            self.assertEqual(current.status, ExecutionStatus.SUCCEEDED, public)
            self.assertIsNotNone(current.result)
            assert current.result is not None
            self.assertEqual(set(current.result.outputs), {"heatmap"})
            scheduled_artifact = current.result.outputs["heatmap"]
            self.assertIsInstance(scheduled_artifact, SharedArtifact)
            assert isinstance(scheduled_artifact, SharedArtifact)
            self.assertEqual(scheduled_artifact.media_type, "image/tiff")

        terminal_response = self.http.get(f"/v1/executions/{execution_id}")
        self.assertEqual(terminal_response.status_code, 200, terminal_response.text)
        self._assert_public_execution(terminal_response.json(), execution_id)
        self.assertEqual(terminal_response.json()["status"], "succeeded")
        _evidence("terminal_http", execution=terminal_response.json())

        with Client(base_url=self.settings.origin, timeout=30) as client:
            recovered = client.workflows.schedule(
                _WORKFLOW,
                request,
                requested_timeout_seconds=self.settings.completion_timeout_seconds,
                idempotency_key=key,
            )
            self.assertFalse(recovered.created)
            self.assertEqual(recovered.execution.id, execution_id)
            self.assertEqual(recovered.execution.status, ExecutionStatus.SUCCEEDED)
            self.assertEqual(recovered.execution.result, current.result)
            immediate = client.workflows.run(
                _WORKFLOW,
                self.settings.sdk_request(f"outputs/{key}-immediate.tiff"),
                workflow_timeout_seconds=self.settings.completion_timeout_seconds,
                timeout=self.settings.completion_timeout_seconds + 30,
            )
        self.assertEqual(set(immediate.outputs), set(current.result.outputs))
        immediate_artifact = immediate.outputs["heatmap"]
        self.assertIsInstance(immediate_artifact, SharedArtifact)
        assert isinstance(immediate_artifact, SharedArtifact)
        self.assertEqual(immediate_artifact.kind, scheduled_artifact.kind)
        self.assertEqual(immediate_artifact.media_type, scheduled_artifact.media_type)
        _evidence(
            "artifact_references",
            execution_id=execution_id,
            scheduled=asdict(scheduled_artifact),
            immediate=asdict(immediate_artifact),
        )
        content = self._compare_artifact_content(
            scheduled_artifact.path, immediate_artifact.path
        )
        _evidence(
            "artifacts",
            execution_id=execution_id,
            scheduled=asdict(scheduled_artifact),
            immediate=asdict(immediate_artifact),
            content=content,
        )

    @unittest.skipUnless(
        os.environ.get("VALIDATION_SCHEDULER_WORK_PAUSED") == "1",
        "root must pause processing claims while keeping HTTP and maintenance alive",
    )
    def test_queued_cancellation_and_replay_remain_terminal(self) -> None:
        """Verify queued cancellation and replay remain terminal.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        from rationai import Client
        from rationai.types.workflows import ExecutionStatus

        key, body, execution_id = self._schedule_queued("cancel", timeout=120)
        with Client(base_url=self.settings.origin, timeout=30) as client:
            cancelled = client.executions.cancel(execution_id)
            self.assertEqual(cancelled.status, ExecutionStatus.CANCELLED)
            self.assertIsNone(cancelled.result)
            repeated = client.executions.cancel(execution_id)
            self.assertEqual(repeated, cancelled)
        with Client(base_url=self.settings.origin, timeout=30) as client:
            polled = client.executions.get(execution_id)
            self.assertEqual(polled, cancelled)
        replay = self.http.post(f"/v1/workflows/{_WORKFLOW}/schedule", json=body)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertFalse(replay.json()["created"])
        self.assertEqual(replay.json()["execution"]["id"], execution_id)
        self.assertEqual(replay.json()["execution"]["status"], "cancelled")
        self._assert_public_execution(replay.json()["execution"], execution_id)
        _evidence(
            "queued_cancelled",
            execution=asdict(polled),
            idempotency_key=key,
            replay=replay.json(),
        )

    @unittest.skipUnless(
        os.environ.get("VALIDATION_SCHEDULER_WORK_PAUSED") == "1",
        "root must pause processing claims while keeping HTTP and maintenance alive",
    )
    def test_queued_deadline_expires_without_processing_and_stays_failed(self) -> None:
        """Verify queued deadline expires without processing and stays failed.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        key, body, execution_id = self._schedule_queued("deadline", timeout=5)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            response = self.http.get(f"/v1/executions/{execution_id}")
            self.assertEqual(response.status_code, 200, response.text)
            execution = response.json()
            self._assert_public_execution(execution, execution_id)
            _evidence("queued_deadline_poll", execution=execution)
            self.assertIn(execution["status"], {"queued", "failed"})
            if execution["status"] == "failed":
                break
            time.sleep(0.5)
        else:
            self.fail("queued execution deadline did not advance within 45 seconds")
        self.assertEqual(execution["error"]["code"], "execution_deadline_exceeded")
        self.assertIsNone(execution["result"])
        replay = self.http.post(f"/v1/workflows/{_WORKFLOW}/schedule", json=body)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertFalse(replay.json()["created"])
        self.assertEqual(replay.json()["execution"], execution)
        cancelled = self.http.post(f"/v1/executions/{execution_id}/cancel")
        self.assertEqual(cancelled.status_code, 200, cancelled.text)
        self.assertEqual(cancelled.json(), execution)
        _evidence(
            "queued_expired",
            execution=execution,
            idempotency_key=key,
            replay=replay.json(),
        )

    def _schedule_queued(
        self, purpose: str, *, timeout: int
    ) -> tuple[str, dict[str, object], str]:
        """Submit standalone work expected to stay queued, register cleanup cancellation, and assert its public admission shape.

        Args:
            purpose: Case label included in the unique live queued-execution identity.
            timeout: Request or scenario time budget in seconds.

        Returns:
            tuple[str, dict[str, object], str]: Selected records, identifiers, or
                fixture values in the ordering described above.
        """
        key = f"native-queued-{purpose}-{uuid4().hex}"
        body = {
            "provider": "standalone",
            "body": self.settings.invocation(f"outputs/{key}.tiff"),
            "requested_timeout_seconds": timeout,
            "idempotency_key": key,
        }
        _evidence("queued_request", request=body)
        response = self.http.post(f"/v1/workflows/{_WORKFLOW}/schedule", json=body)
        self.assertEqual(response.status_code, 202, response.text)
        admitted = response.json()
        self.assertTrue(admitted["created"])
        execution = admitted["execution"]
        execution_id = execution["id"]
        self.addCleanup(self._cancel_unfinished, execution_id)
        self._assert_public_execution(execution, execution_id)
        self.assertEqual(
            execution["status"], "queued", "processing claims are not paused"
        )
        return key, body, execution_id

    def _assert_public_execution(
        self, execution: dict[str, object], execution_id: str
    ) -> None:
        """Require only public fields and hide results/errors until their matching terminal outcome.

        Args:
            execution: Current execution snapshot or public view to inspect or
                transition.
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            None: No return value.
        """
        self.assertEqual(
            set(execution), {"id", "workflow_id", "status", "result", "error"}
        )
        self.assertEqual(execution["id"], execution_id)
        self.assertEqual(execution["workflow_id"], _WORKFLOW)
        self.assertIn(execution["status"], {"queued", "running", *_TERMINAL})
        if execution["status"] != "succeeded":
            self.assertIsNone(execution["result"])
        if execution["status"] != "failed":
            self.assertIsNone(execution["error"])

    def _cancel_unfinished(self, execution_id: str) -> None:
        """Cancel leftover live executions during cleanup, recording transport unavailability without masking the test outcome.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            None: No return value.
        """
        try:
            response = self.http.get(f"/v1/executions/{execution_id}")
            if (
                response.status_code == 200
                and response.json()["status"] not in _TERMINAL
            ):
                cancelled = self.http.post(f"/v1/executions/{execution_id}/cancel")
                _evidence(
                    "cleanup_cancel",
                    execution_id=execution_id,
                    status_code=cancelled.status_code,
                )
        except httpx.TransportError as error:
            _evidence(
                "cleanup_unavailable",
                execution_id=execution_id,
                error=type(error).__name__,
            )

    def _compare_artifact_content(
        self, scheduled_path: str, immediate_path: str
    ) -> dict[str, object]:
        """If local shared-file access is configured, verify contained nonempty files and compare scheduled/immediate SHA-256 hashes.

        Args:
            scheduled_path: Artifact reference returned by durable scheduled execution.
            immediate_path: Artifact reference returned by immediate execution.

        Returns:
            dict[str, object]: Decoded, serialized, or recorded fields described above.
        """
        if self.settings.artifact_root is None:
            return {"verified": False, "reason": "requires separate shared-file access"}
        paths = []
        for remote_path in (scheduled_path, immediate_path):
            relative = PurePosixPath(remote_path).relative_to(
                self.settings.storage_root
            )
            local_path = self.settings.artifact_root.joinpath(*relative.parts).resolve(
                strict=True
            )
            self.assertTrue(local_path.is_relative_to(self.settings.artifact_root))
            self.assertTrue(local_path.is_file())
            self.assertGreater(local_path.stat().st_size, 0)
            paths.append(local_path)
        hashes = []
        for path in paths:
            with path.open("rb") as artifact:
                hashes.append(hashlib.file_digest(artifact, "sha256").hexdigest())
        self.assertEqual(hashes[0], hashes[1], "completed TIFF contents differ")
        return {"verified": True, "sha256": hashes[0], "bytes": paths[0].stat().st_size}


if __name__ == "__main__":
    unittest.main()
