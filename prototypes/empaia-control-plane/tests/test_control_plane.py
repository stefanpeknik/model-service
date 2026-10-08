"""Check historical prototype SQLite admission, attempt ownership, timeout, and cancellation rules.

These cases describe the archived SQLite prototype rather than the current production
scheduler.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from control_plane.domain import (
    AdmissionRequest,
    AppMapping,
    AttemptOwnershipError,
    ExecutionSpec,
    ExecutionState,
    InvalidTransitionError,
    SubmissionConflictError,
    TimeoutKind,
)
from control_plane.registry import AppRegistry
from control_plane.service import ExecutionService, UnknownAppMappingError
from control_plane.store import SQLiteExecutionStore


class Clock:
    """Provide a manually advanced UTC clock for deterministic prototype lifecycle tests."""

    def __init__(self) -> None:
        """Start the prototype test clock at its fixed UTC baseline."""
        self.current = datetime(2026, 9, 6, tzinfo=UTC)

    def __call__(self) -> datetime:
        """Return the current simulated UTC timestamp.

        Returns:
            datetime: Timezone-aware timestamp used for deadlines or durable lifecycle
                decisions.
        """
        return self.current

    def advance(self, seconds: int) -> None:
        """Advance the prototype test clock by the supplied number of seconds.

        Args:
            seconds: Number of seconds to advance the prototype test clock.

        Returns:
            None: No return value.
        """
        self.current += timedelta(seconds=seconds)


class ControlPlaneTestCase(unittest.TestCase):
    """Exercise historical prototype SQLite admission, attempt ownership, timeout, and cancellation rules."""

    def setUp(self) -> None:
        """Create isolated fixtures and clients for historical prototype SQLite admission, attempt ownership, timeout, and cancellation rules.

        Returns:
            None: No return value.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.database_path = Path(self.temp_dir.name) / "jobs.db"
        self.clock = Clock()
        self.store = SQLiteExecutionStore(self.database_path, clock=self.clock)
        self.mapping = AppMapping(
            app_id="prostate-app",
            app_version="1",
            model_id="prostate-classifier",
            model_version="1",
            workflow="wsi-to-classification-heatmap",
            resource_profile="mig20",
            max_runtime_seconds=14_400,
            model_parameter_names=("threshold",),
        )
        self.service = ExecutionService(
            self.store, AppRegistry([self.mapping]), installation_alias="poc"
        )
        self.request = AdmissionRequest(
            installation_alias="poc",
            empaia_job_id="job-123",
            registered_app_id="prostate-app",
            registered_app_version="1",
            idempotency_key="poc:job-123:prostate-app:1",
        )
        self.spec = ExecutionSpec(
            model_id="prostate-classifier",
            model_version="1",
            workflow="wsi-to-classification-heatmap",
            resource_profile="mig20",
            max_runtime_seconds=14_400,
            wsi_id="wsi-123",
            storage_id="slides",
            relative_locator="case/slide.mrxs",
            output_name="heatmap",
            model_parameters={"threshold": 0.5},
        )

    def ready_execution(self) -> str:
        """Admit, resolve, and mark the prototype fixture ready, returning its durable ID.

        Returns:
            str: Durable ID of the admitted and resolved prototype fixture.
        """
        execution = self.service.admit(self.request).execution
        self.store.begin_resolution(execution.id)
        self.store.mark_ready(execution.id, self.spec)
        return execution.id

    def test_admission_is_durable_and_idempotent(self) -> None:
        """Verify admission is durable and idempotent.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        first = self.service.admit(self.request)
        duplicate = self.service.admit(self.request)
        reopened = SQLiteExecutionStore(self.database_path).get(first.execution.id)

        self.assertTrue(first.created)
        self.assertFalse(duplicate.created)
        self.assertEqual(first.execution.id, duplicate.execution.id)
        self.assertEqual("prostate-classifier", reopened.model_id)
        self.assertEqual(3, reopened.max_attempts)

    def test_conflicting_submission_identity_or_key_is_rejected(self) -> None:
        """Verify conflicting submission identity or key is rejected.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        self.service.admit(self.request)
        different_job_same_key = AdmissionRequest(
            installation_alias="poc",
            empaia_job_id="different-job",
            registered_app_id="prostate-app",
            registered_app_version="1",
            idempotency_key=self.request.idempotency_key,
        )
        same_job_different_key = AdmissionRequest(
            installation_alias="poc",
            empaia_job_id=self.request.empaia_job_id,
            registered_app_id="prostate-app",
            registered_app_version="1",
            idempotency_key="poc:job-123:unexpected-key",
        )

        with self.assertRaises(SubmissionConflictError):
            self.service.admit(different_job_same_key)
        with self.assertRaises(SubmissionConflictError):
            self.service.admit(same_job_different_key)

    def test_unknown_mapping_is_rejected_without_creating_a_job(self) -> None:
        """Verify unknown mapping is rejected without creating a job.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        request = AdmissionRequest(
            installation_alias="poc",
            empaia_job_id="unknown-job",
            registered_app_id="unknown-app",
            registered_app_version="1",
            idempotency_key="poc:unknown-job:unknown-app:1",
        )

        with self.assertRaises(UnknownAppMappingError):
            self.service.admit(request)

    def test_claim_is_retry_idempotent_and_owned_by_one_attempt(self) -> None:
        """Verify claim is retry idempotent and owned by one attempt.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        execution_id = self.ready_execution()
        attempt_id = str(uuid4())
        claimed = self.store.claim(execution_id, attempt_id)

        self.assertEqual(claimed, self.store.claim(execution_id, attempt_id))
        with self.assertRaises(InvalidTransitionError):
            self.store.claim(execution_id, str(uuid4()))

        heartbeat = self.store.heartbeat_attempt(execution_id, attempt_id)
        self.assertEqual(attempt_id, heartbeat.current_attempt_id)

    def test_running_cancellation_is_fenced_without_waiting_for_heartbeat_loss(
        self,
    ) -> None:
        """Verify running cancellation is fenced without waiting for heartbeat loss.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        execution_id = self.ready_execution()
        attempt_id = str(uuid4())
        self.store.claim(execution_id, attempt_id)
        self.service.cancel(execution_id)

        with self.assertRaises(InvalidTransitionError):
            self.store.claim(execution_id, str(uuid4()))
        self.clock.advance(14_400)
        self.assertEqual([], self.store.expire_timeouts(None))
        cancelled = self.store.get(execution_id)
        self.assertEqual(ExecutionState.CANCELLED, cancelled.state)
        self.assertIsNone(cancelled.current_attempt_id)

    def test_timeout_policy_distinguishes_capacity_from_model_runtime(self) -> None:
        """Verify timeout policy distinguishes capacity from model runtime.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        capacity_execution_id = self.ready_execution()
        self.clock.advance(60)
        self.assertEqual([], self.store.expire_timeouts(None))
        capacity_expired = self.store.claim(capacity_execution_id, str(uuid4()), 60)
        self.assertEqual(TimeoutKind.INITIAL_CAPACITY, capacity_expired.timeout_kind)

        second_request = AdmissionRequest(
            installation_alias="poc",
            empaia_job_id="job-runtime",
            registered_app_id="prostate-app",
            registered_app_version="1",
            idempotency_key="poc:job-runtime:prostate-app:1",
        )
        runtime_execution = self.service.admit(second_request).execution
        self.store.begin_resolution(runtime_execution.id)
        self.store.mark_ready(runtime_execution.id, self.spec)
        attempt_id = str(uuid4())
        self.store.claim(runtime_execution.id, attempt_id)
        self.clock.advance(14_400)
        runtime_expired = self.store.heartbeat_attempt(runtime_execution.id, attempt_id)

        self.assertEqual(capacity_execution_id, capacity_expired.id)
        self.assertEqual(TimeoutKind.MODEL_RUNTIME, runtime_expired.timeout_kind)

    def test_worker_retry_does_not_reset_original_runtime_deadline(self) -> None:
        """Verify worker retry does not reset original runtime deadline.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        short_mapping = AppMapping(
            app_id="prostate-app",
            app_version="1",
            model_id="prostate-classifier",
            model_version="1",
            workflow="wsi-to-classification-heatmap",
            resource_profile="mig20",
            max_runtime_seconds=10,
            model_parameter_names=("threshold",),
        )
        service = ExecutionService(
            self.store, AppRegistry([short_mapping]), installation_alias="poc"
        )
        execution = service.admit(self.request).execution
        self.store.begin_resolution(execution.id)
        self.store.mark_ready(execution.id, replace(self.spec, max_runtime_seconds=10))
        first = self.store.claim(execution.id, str(uuid4()))
        self.clock.advance(4)
        renewed = self.store.claim(
            execution.id,
            first.current_attempt_id or self.fail("claim had no attempt"),
        )
        self.assertEqual(self.clock.current, renewed.attempt_heartbeat_at)
        self.store.recover_attempt(
            execution.id,
            first.current_attempt_id or self.fail("claim had no attempt"),
            self.clock.current,
            self.clock.current,
        )
        with self.assertRaises(AttemptOwnershipError):
            self.store.claim(execution.id, first.current_attempt_id or "")
        second = self.store.claim(execution.id, str(uuid4()))
        self.clock.advance(6)

        expired = self.store.expire_timeouts(None)

        self.assertEqual(TimeoutKind.MODEL_RUNTIME, expired[0].timeout_kind)
        self.assertEqual(first.first_claimed_at, second.first_claimed_at)

    def test_spec_cannot_override_profile_or_persist_credentials(self) -> None:
        """Verify spec cannot override profile or persist credentials.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        execution = self.service.admit(self.request).execution
        self.store.begin_resolution(execution.id)
        conflicting = ExecutionSpec(
            model_id="other-model",
            model_version=self.spec.model_version,
            workflow=self.spec.workflow,
            resource_profile=self.spec.resource_profile,
            max_runtime_seconds=self.spec.max_runtime_seconds,
            wsi_id=self.spec.wsi_id,
            storage_id=self.spec.storage_id,
            relative_locator=self.spec.relative_locator,
            output_name=self.spec.output_name,
            model_parameters={},
        )

        with self.assertRaises(InvalidTransitionError):
            self.store.mark_ready(execution.id, conflicting)
        with self.assertRaisesRegex(ValueError, "credentials"):
            ExecutionSpec(
                model_id=self.spec.model_id,
                model_version=self.spec.model_version,
                workflow=self.spec.workflow,
                resource_profile=self.spec.resource_profile,
                max_runtime_seconds=self.spec.max_runtime_seconds,
                wsi_id=self.spec.wsi_id,
                storage_id=self.spec.storage_id,
                relative_locator=self.spec.relative_locator,
                output_name=self.spec.output_name,
                model_parameters={"token": "must-not-persist"},
            )


if __name__ == "__main__":
    unittest.main()
